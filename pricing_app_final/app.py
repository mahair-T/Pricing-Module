import requests  # <--- Add this at the top
import folium
from streamlit_folium import st_folium
###
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import pickle
import os
import io
import re
import json
from datetime import datetime
import itertools

# Fuzzy matching for city name resolution
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False 

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="Freight Pricing Tool", page_icon="🚚", layout="wide")

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================
# 🔧 CONFIGURATION
# ============================================
# MASTER GRID SHEET URL
BULK_PRICING_SHEET_URL = "https://docs.google.com/spreadsheets/d/1u4qyqE626mor0OV1JHYO2Ejd5chmPy3wi1P0AWdhLPw/edit?gid=0#gid=0"
BULK_TAB_NAME = "All Lanes"

# ============================================
# PROVINCE TO REGION MAPPING
# Maps Saudi Arabia's 13 official provinces to 5 freight regions
# ============================================
GEOJSON_PROVINCE_TO_STANDARD = {
    'Ash Sharqiyah': 'Eastern Province',
    'Al Hudud ash Shamaliyah': 'Northern Borders Province',
    'Al Jawf': 'Al Jouf Province',
    'Al Madinah': 'Madinah Province',
    'Al Quassim': 'Qassim Province',
    'Ar Riyad': 'Riyadh Province',
    "Ha'il": 'Hail Province',
    'Jizan': 'Jazan Province',
    'Makkah': 'Makkah Province',
    'Najran': 'Najran Province',
    'Tabuk': 'Tabuk Province',
    '`Asir': 'Asir Province',
    'Al Bahah': 'Al Bahah Province',
}

PROVINCE_TO_REGION = {
    'Makkah Province': 'Western',
    'Eastern Province': 'Eastern',
    'Riyadh Province': 'Central',
    'Asir Province': 'Southern',
    'Qassim Province': 'Northern',
    'Jazan Province': 'Southern',
    'Madinah Province': 'Western',
    'Al Bahah Province': 'Southern',
    'Najran Province': 'Southern',
    'Hail Province': 'Northern',
    'Tabuk Province': 'Northern',
    'Al Jouf Province': 'Northern',
    'Northern Borders Province': 'Northern'
}

# ============================================
# GEOJSON LOADING FOR PROVINCE DETECTION
# ============================================
@st.cache_resource
def load_province_geojson():
    """
    Load Saudi Arabia province boundaries from GeoJSON for coordinate-based province detection.
    Returns list of (province_name, polygon) tuples for point-in-polygon checks.
    """
    try:
        from shapely.geometry import shape, Point
        
        geojson_path = os.path.join(APP_DIR, 'model_export', 'saudi_regions_enhanced.geojson')
        if not os.path.exists(geojson_path):
            # Try alternate location
            geojson_path = os.path.join(APP_DIR, 'saudi_regions_enhanced.geojson')
        
        if not os.path.exists(geojson_path):
            return None
        
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        provinces = []
        for feature in geojson_data.get('features', []):
            props = feature.get('properties', {})
            geojson_name = props.get('name', '')
            
            # Map GeoJSON province name to standard province name
            province_name = GEOJSON_PROVINCE_TO_STANDARD.get(geojson_name, geojson_name)
            
            geometry = feature.get('geometry')
            if geometry:
                try:
                    polygon = shape(geometry)
                    provinces.append((province_name, polygon))
                except Exception:
                    pass
        
        return provinces
    except ImportError:
        # shapely not available
        return None
    except Exception as e:
        return None

PROVINCE_POLYGONS = load_province_geojson()

def get_province_from_coordinates(lat, lon):
    """
    Determine province from latitude/longitude coordinates using GeoJSON boundaries.
    
    Args:
        lat: Latitude (float)
        lon: Longitude (float)
    
    Returns:
        (province_name, region_name) or (None, None) if not found
    """
    if PROVINCE_POLYGONS is None:
        return None, None
    
    try:
        from shapely.geometry import Point
        
        point = Point(lon, lat)  # Note: shapely uses (lon, lat) order
        
        for province_name, polygon in PROVINCE_POLYGONS:
            if polygon.contains(point):
                region = PROVINCE_TO_REGION.get(province_name)
                return province_name, region
        
        return None, None
    except Exception:
        return None, None

def normalize_english_text(text):
    """Normalize English text for lookups - lowercase, strip, normalize whitespace/hyphens."""
    if pd.isna(text) or text is None:
        return None
    text = str(text).strip().lower()
    text = re.sub(r'[-_]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def normalize_aggressive(text):
    """
    Aggressive normalization for fuzzy-ish matching.
    Removes ALL non-alphanumeric characters, lowercases, strips diacritics from Arabic.
    
    Examples:
    - "Al-Jubail" -> "aljubail"
    - "Al Jubail" -> "aljubail"  
    - "ALJUBAIL" -> "aljubail"
    - "Hafar Al Batin" -> "hafaralbatin"
    - "Hafar Al-Batin" -> "hafaralbatin"
    """
    if pd.isna(text) or text is None:
        return None
    text = str(text).strip().lower()
    # Remove all non-alphanumeric characters (keeps Arabic letters too)
    text = re.sub(r'[^a-z0-9\u0600-\u06ff]', '', text)
    return text if text else None

# ============================================
# ERROR LOGGING (Google Sheets) - RESTORED ORIGINAL
# ============================================
def get_gsheet_client():
    """Get Google Sheets client using Streamlit secrets."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        
        if 'gcp_service_account' not in st.secrets:
            return None
        
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        credentials = Credentials.from_service_account_info(
            st.secrets['gcp_service_account'],
            scopes=scopes
        )
        
        return gspread.authorize(credentials)
    except Exception as e:
        return None

@st.cache_resource
def get_error_sheet():
    """Get or create the error log sheet."""
    try:
        client = get_gsheet_client()
        if client is None:
            return None
        
        sheet_url = st.secrets.get('error_log_sheet_url')
        if not sheet_url:
            return None
        
        spreadsheet = client.open_by_url(sheet_url)
        
        try:
            worksheet = spreadsheet.worksheet('ErrorLog')
        except:
            worksheet = spreadsheet.add_worksheet(title='ErrorLog', rows=1000, cols=10)
            worksheet.update('A1:G1', [['Timestamp', 'Type', 'Pickup_City', 'Destination_City', 'Pickup_EN', 'Destination_EN', 'Details']])
        
        return worksheet
    except Exception as e:
        return None

@st.cache_resource
def get_reported_sheet():
    """Get or create the Reported sheet for user-reported issues."""
    try:
        client = get_gsheet_client()
        if client is None:
            return None
        
        sheet_url = st.secrets.get('error_log_sheet_url')
        if not sheet_url:
            return None
        
        spreadsheet = client.open_by_url(sheet_url)
        
        try:
            worksheet = spreadsheet.worksheet('Reported')
        except:
            worksheet = spreadsheet.add_worksheet(title='Reported', rows=1000, cols=10)
            worksheet.update('A1:H1', [['Timestamp', 'Pickup_City', 'Destination_City', 'Pickup_EN', 'Destination_EN', 'Current_Distance', 'Issue', 'User_Notes']])
        
        return worksheet
    except Exception as e:
        return None

def get_matched_distances_sheet():
    """
    Get or create the MatchedDistances sheet for Google Maps API distance lookups.
    NOT cached because we need fresh connection for reads/writes.
    """
    try:
        client = get_gsheet_client()
        if client is None:
            return None
        
        sheet_url = st.secrets.get('error_log_sheet_url')
        if not sheet_url:
            return None
        
        spreadsheet = client.open_by_url(sheet_url)
        
        try:
            worksheet = spreadsheet.worksheet('MatchedDistances')
        except:
            # Create sheet with headers and formula template
            worksheet = spreadsheet.add_worksheet(title='MatchedDistances', rows=1000, cols=10)
            headers = ['Timestamp', 'Pickup_AR', 'Destination_AR', 'Pickup_EN', 'Destination_EN', 
                       'Distance_Formula', 'Distance_Value', 'Status', 'Added_To_Pickle']
            worksheet.update('A1:I1', [headers])
        
        return worksheet
    except Exception as e:
        return None

def log_missing_distance_for_lookup(pickup_ar, dest_ar, pickup_en, dest_en, immediate=True):
    """
    Log a missing distance to the MatchedDistances sheet for Google Maps lookup.
    Only logs if this city pair isn't already in the sheet or pending queue.
    
    Args:
        pickup_ar, dest_ar: Arabic city names
        pickup_en, dest_en: English city names
        immediate: If True, write to sheet immediately. If False, queue for batch write.
    """
    # Initialize pending queue if needed
    if 'matched_distances_pending' not in st.session_state:
        st.session_state.matched_distances_pending = []
    
    # Check if already in pending queue (both directions)
    for item in st.session_state.matched_distances_pending:
        if (item['pickup_ar'] == pickup_ar and item['dest_ar'] == dest_ar) or \
           (item['pickup_ar'] == dest_ar and item['dest_ar'] == pickup_ar):
            return False  # Already queued
    
    if immediate:
        # Write immediately (for single-lane pricing)
        try:
            worksheet = get_matched_distances_sheet()
            if not worksheet:
                return False
            
            # Check if this pair already exists in sheet (check both directions)
            existing = worksheet.get_all_values()
            for row in existing[1:]:  # Skip header
                if len(row) >= 5:
                    if (row[1] == pickup_ar and row[2] == dest_ar) or \
                       (row[1] == dest_ar and row[2] == pickup_ar):
                        return False  # Already exists
            
            # Find next row
            next_row = len(existing) + 1
        
            # 1. Build the GOOGLEMAPS_DISTANCE formula for Column F
            formula = f'=GOOGLEMAPS_DISTANCE("{pickup_en}, Saudi Arabia", "{dest_en}, Saudi Arabia", "driving")'
            
            # 2. Build the Logic Formula for Column G
            # Logic: IF(F is a number greater than 0, return F, else blank)
            value_ref = f'=IF(N(F{next_row})>0, F{next_row}, "")'
            
            # Write the row
            row_data = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                pickup_ar,
                dest_ar,
                pickup_en,
                dest_en,
                formula,    # Column F: The calculation
                value_ref,  # Column G: The conditional check
                'Pending',
                'No'
            ]
            worksheet.update(f'A{next_row}:I{next_row}', [row_data], value_input_option='USER_ENTERED')
            return True
        except Exception as e:
            return False
    else:
        # Queue for batch write (for bulk pricing)
        st.session_state.matched_distances_pending.append({
            'pickup_ar': pickup_ar,
            'dest_ar': dest_ar,
            'pickup_en': pickup_en,
            'dest_en': dest_en
        })
        return True

def get_matched_cities_sheet():
    """
    Get or create the MatchedCities sheet for logging city match resolutions.
    Tracks fuzzy matches and new city additions from bulk uploads.
    """
    try:
        client = get_gsheet_client()
        if client is None:
            return None
        
        sheet_url = st.secrets.get('error_log_sheet_url')
        if not sheet_url:
            return None
        
        spreadsheet = client.open_by_url(sheet_url)
        
        try:
            worksheet = spreadsheet.worksheet('MatchedCities')
        except:
            # Create sheet with headers
            worksheet = spreadsheet.add_worksheet(title='MatchedCities', rows=1000, cols=12)
            headers = ['Timestamp', 'Original_Input', 'Matched_Canonical', 'Match_Type', 
                       'Confidence', 'Latitude', 'Longitude', 'Province', 'Region',
                       'Added_To_CSV', 'User', 'Source']
            worksheet.update('A1:L1', [headers])
        
        return worksheet
    except Exception as e:
        return None

def get_append_sheet():
    """
    Get or create the Append sheet for easy CSV additions.
    This sheet mirrors the city_normalization CSV structure for manual appending.
    """
    try:
        client = get_gsheet_client()
        if client is None:
            return None
        
        sheet_url = st.secrets.get('error_log_sheet_url')
        if not sheet_url:
            return None
        
        spreadsheet = client.open_by_url(sheet_url)
        
        try:
            worksheet = spreadsheet.worksheet('Append')
        except:
            # Create sheet with headers matching city_normalization CSV
            worksheet = spreadsheet.add_worksheet(title='Append', rows=1000, cols=10)
            headers = ['variant', 'canonical', 'region', 'province', 'latitude', 'longitude', 
                       'source', 'timestamp', 'user', 'added_to_csv']
            worksheet.update('A1:J1', [headers])
        
        return worksheet
    except Exception as e:
        return None

def log_city_match(original_input, matched_canonical, match_type, confidence, 
                   latitude=None, longitude=None, province=None, region=None, 
                   user='', source='bulk_upload', immediate=True):
    """
    Log a city match resolution to the MatchedCities sheet.
    
    Args:
        original_input: The original unmatched city name
        matched_canonical: The canonical name it was matched/assigned to
        match_type: 'Fuzzy Match Confirmed' or 'New City'
        confidence: Match confidence score (0-100 for fuzzy, N/A for new)
        latitude, longitude: Coordinates (for new cities)
        province: Province name
        region: Freight region (Eastern, Western, Central, Northern, Southern)
        user: Username who made the resolution
        source: Source of the match (bulk_upload, manual, etc.)
        immediate: If True, write immediately. If False, queue for batch.
    """
    # Initialize pending queue if needed
    if 'matched_cities_pending' not in st.session_state:
        st.session_state.matched_cities_pending = []
    
    row_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'original_input': original_input,
        'matched_canonical': matched_canonical,
        'match_type': match_type,
        'confidence': confidence if confidence else 'N/A',
        'latitude': latitude if latitude else '',
        'longitude': longitude if longitude else '',
        'province': province if province else '',
        'region': region if region else '',
        'added_to_csv': 'No',
        'user': user,
        'source': source
    }
    
    if immediate:
        try:
            worksheet = get_matched_cities_sheet()
            if worksheet:
                row = [
                    row_data['timestamp'], row_data['original_input'], row_data['matched_canonical'],
                    row_data['match_type'], str(row_data['confidence']), str(row_data['latitude']),
                    str(row_data['longitude']), row_data['province'], row_data['region'],
                    row_data['added_to_csv'], row_data['user'], row_data['source']
                ]
                worksheet.append_row(row)
                return True
        except Exception as e:
            pass
        return False
    else:
        st.session_state.matched_cities_pending.append(row_data)
        return True

def log_to_append_sheet(variant, canonical, region, province=None, latitude=None, longitude=None,
                        source='bulk_upload', user='', immediate=True):
    """
    Log a new city variant to the Append sheet for CSV addition.
    
    Args:
        variant: The variant name (as entered by user)
        canonical: The canonical name
        region: Freight region
        province: Province name (optional)
        latitude, longitude: Coordinates (optional)
        source: Source of the entry
        user: Username
        immediate: If True, write immediately
    """
    if 'append_sheet_pending' not in st.session_state:
        st.session_state.append_sheet_pending = []
    
    row_data = {
        'variant': variant,
        'canonical': canonical,
        'region': region,
        'province': province if province else '',
        'latitude': latitude if latitude else '',
        'longitude': longitude if longitude else '',
        'source': source,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user': user,
        'added_to_csv': 'No'
    }
    
    if immediate:
        try:
            worksheet = get_append_sheet()
            if worksheet:
                row = [
                    row_data['variant'], row_data['canonical'], row_data['region'],
                    row_data['province'], str(row_data['latitude']), str(row_data['longitude']),
                    row_data['source'], row_data['timestamp'], row_data['user'], row_data['added_to_csv']
                ]
                worksheet.append_row(row)
                return True
        except Exception as e:
            pass
        return False
    else:
        st.session_state.append_sheet_pending.append(row_data)
        return True

def flush_matched_cities_to_sheet():
    """Flush pending city matches to Google Sheet with Grid Expansion."""
    if 'matched_cities_pending' not in st.session_state or len(st.session_state.matched_cities_pending) == 0:
        return True, 0
    
    pending = st.session_state.matched_cities_pending
    try:
        worksheet = get_matched_cities_sheet()
        if worksheet:
            rows = []
            for item in pending:
                rows.append([
                    item['timestamp'], item['original_input'], item['matched_canonical'],
                    item['match_type'], str(item['confidence']), str(item['latitude']),
                    str(item['longitude']), item['province'], item['region'],
                    item['added_to_csv'], item['user'], item['source']
                ])
            if rows:
                # ✅ FIX: Force-add rows to prevent "Grid Limits" error
                worksheet.add_rows(len(rows))
                worksheet.append_rows(rows)
                
            st.session_state.matched_cities_pending = []
            return True, len(rows)
    except Exception as e:
        if 'flush_errors' not in st.session_state:
            st.session_state.flush_errors = []
        st.session_state.flush_errors.append(f"City Flush Error: {str(e)}")
        pass
    return False, 0

def flush_append_sheet():
    """Flush pending append entries to Google Sheet with Grid Expansion."""
    if 'append_sheet_pending' not in st.session_state or len(st.session_state.append_sheet_pending) == 0:
        return True, 0
    
    pending = st.session_state.append_sheet_pending
    try:
        worksheet = get_append_sheet()
        if worksheet:
            rows = []
            for item in pending:
                rows.append([
                    item['variant'], item['canonical'], item['region'],
                    item['province'], str(item['latitude']), str(item['longitude']),
                    item['source'], item['timestamp'], item['user'], item['added_to_csv']
                ])
            if rows:
                # ✅ FIX: Force-add rows to prevent "Grid Limits" error
                worksheet.add_rows(len(rows))
                worksheet.append_rows(rows)
                
            st.session_state.append_sheet_pending = []
            return True, len(rows)
    except Exception as e:
        if 'flush_errors' not in st.session_state:
            st.session_state.flush_errors = []
        st.session_state.flush_errors.append(f"Append Flush Error: {str(e)}")
        pass
    return False, 0

def flush_matched_distances_to_sheet():
    """
    Flush pending matched distances to Google Sheet.
    Automatically resizes the sheet if more rows are needed.
    """
    if 'matched_distances_pending' not in st.session_state or len(st.session_state.matched_distances_pending) == 0:
        return True, 0
    
    pending = st.session_state.matched_distances_pending
    
    try:
        worksheet = get_matched_distances_sheet()
        if not worksheet:
            if 'flush_errors' not in st.session_state: st.session_state.flush_errors = []
            st.session_state.flush_errors.append(f"Could not get MatchedDistances worksheet")
            return False, 0
        
        # Get existing data to check duplicates
        existing = worksheet.get_all_values()
        existing_pairs = set()
        for row in existing[1:]:
            if len(row) >= 5:
                existing_pairs.add((row[1], row[2]))
                existing_pairs.add((row[2], row[1]))  # Both directions
        
        next_row = len(existing) + 1
        rows_to_add = []
        
        for item in pending:
            # Skip if already exists
            if (item['pickup_ar'], item['dest_ar']) in existing_pairs:
                continue
            if (item['dest_ar'], item['pickup_ar']) in existing_pairs:
                continue
            
            # Build formulas
            formula = f'=GOOGLEMAPS_DISTANCE("{item["pickup_en"]}, Saudi Arabia", "{item["dest_en"]}, Saudi Arabia", "driving")'
            value_ref = f'=IF(N(F{next_row})>0, F{next_row}, "")'
            
            row_data = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                item['pickup_ar'],
                item['dest_ar'],
                item['pickup_en'],
                item['dest_en'],
                formula,
                value_ref,
                'Pending',
                'No'
            ]
            rows_to_add.append(row_data)
            existing_pairs.add((item['pickup_ar'], item['dest_ar']))
            next_row += 1
        
        if rows_to_add:
            # --- CRITICAL FIX: RESIZE SHEET IF NEEDED ---
            current_row_count = worksheet.row_count
            required_rows = next_row + 10 # Add a small buffer
            
            if required_rows > current_row_count:
                # Add enough rows to fit data plus a buffer of 500 for future
                rows_to_append = max(len(rows_to_add), 500)
                worksheet.add_rows(rows_to_append)
            # --------------------------------------------

            # Write all rows at once
            start_row = len(existing) + 1
            end_row = start_row + len(rows_to_add) - 1
            worksheet.update(f'A{start_row}:I{end_row}', rows_to_add, value_input_option='USER_ENTERED')
        
        # Clear pending queue
        st.session_state.matched_distances_pending = []
        return True, len(rows_to_add)
    
    except Exception as e:
        if 'flush_errors' not in st.session_state: st.session_state.flush_errors = []
        st.session_state.flush_errors.append(f"Flush Error: {str(e)}")
        return False, 0
        
def update_city_normalization_pickle(new_entries):
    """
    Update the in-memory city normalization dictionaries with new entries.
    This allows new cities to be used immediately without restarting the app.
    
    Args:
        new_entries: List of dicts with keys: variant, canonical, region, province, latitude, longitude
    
    Returns: (success, count_added)
    """
    global CITY_TO_CANONICAL, CITY_TO_CANONICAL_LOWER, CITY_TO_CANONICAL_AGGRESSIVE
    global CITY_TO_REGION, CANONICAL_TO_REGION, CITY_AR_TO_EN, CITY_EN_TO_AR
    global CITY_TO_PROVINCE, CANONICAL_TO_PROVINCE, FUZZY_VARIANTS_LIST
    
    added_count = 0
    
    for entry in new_entries:
        variant = entry.get('variant', '').strip()
        canonical = entry.get('canonical', '').strip()
        region = entry.get('region', '')
        province = entry.get('province', '')
        
        if not variant or not canonical:
            continue
        
        # Add to exact match lookup
        CITY_TO_CANONICAL[variant] = canonical
        
        # Add to lowercase lookup
        variant_lower = normalize_english_text(variant)
        if variant_lower:
            CITY_TO_CANONICAL_LOWER[variant_lower] = canonical
        
        # Add to aggressive lookup
        variant_aggressive = normalize_aggressive(variant)
        if variant_aggressive:
            CITY_TO_CANONICAL_AGGRESSIVE[variant_aggressive] = canonical
        
        # Add region mapping
        if region:
            CITY_TO_REGION[variant] = region
            if canonical not in CANONICAL_TO_REGION:
                CANONICAL_TO_REGION[canonical] = region
        
        # Add province mapping
        if province:
            CITY_TO_PROVINCE[variant] = province
            if canonical not in CANONICAL_TO_PROVINCE:
                CANONICAL_TO_PROVINCE[canonical] = province
        
        # Update English display names if variant is English
        if re.match(r'^[A-Za-z\s\-\(\)\.\']+$', variant):
            if canonical not in CITY_AR_TO_EN:
                CITY_AR_TO_EN[canonical] = variant
                CITY_EN_TO_AR[variant] = canonical
        
        added_count += 1
    
    # Rebuild fuzzy variants list
    if RAPIDFUZZ_AVAILABLE:
        FUZZY_VARIANTS_LIST = get_all_variants_for_fuzzy_matching()
    
    return True, added_count

def get_resolved_distances_from_sheet():
    """
    Read resolved distances from the MatchedDistances sheet.
    Returns list of dicts with pickup_ar, dest_ar, distance for rows with valid distances.
    Also includes 'is_suggestion' flag to indicate if it's a manual user suggestion.
    """
    try:
        worksheet = get_matched_distances_sheet()
        if not worksheet:
            return []
        
        all_data = worksheet.get_all_values()
        resolved = []
        
        for i, row in enumerate(all_data[1:], start=2):  # Skip header, track row number
            # Need at least 7 columns (up to column G with distance value)
            if len(row) >= 7:
                pickup_ar = row[1] if len(row) > 1 else ''
                dest_ar = row[2] if len(row) > 2 else ''
                distance_value = row[6] if len(row) > 6 else ''  # Column G - Distance_Value
                status = row[7] if len(row) > 7 else ''  # Column H - Status
                added_to_pickle = row[8] if len(row) > 8 else 'No'  # Column I - default to 'No' if missing
                
                # Check if this is a manual user suggestion
                is_suggestion = 'Suggested by' in status
                
                # Only get rows with valid numeric distance that haven't been added yet
                if distance_value and added_to_pickle != 'Yes' and pickup_ar and dest_ar:
                    try:
                        # Handle various number formats
                        dist_clean = str(distance_value).replace(',', '').replace(' km', '').replace('km', '').strip()
                        # Handle float strings like "960.0"
                        dist_km = float(dist_clean)
                        if dist_km > 0:
                            resolved.append({
                                'row': i,
                                'pickup_ar': pickup_ar,
                                'dest_ar': dest_ar,
                                'distance_km': dist_km,
                                'is_suggestion': is_suggestion
                            })
                    except (ValueError, TypeError):
                        pass
        
        return resolved
    except Exception as e:
        return []

def get_failed_distances_from_sheet():
    """
    Read FAILED distance lookups from the MatchedDistances sheet.
    These are rows where Google Maps failed to find a distance (Column G has an error or invalid value).
    
    Returns list of dicts with row info for rows that need manual distance input.
    """
    # Common Google Sheets error values
    ERROR_VALUES = {'#N/A', '#ERROR!', '#VALUE!', '#REF!', '#NAME?', '#DIV/0!', '#NULL!', 
                    'Loading...', 'loading...', 'Error', 'error', 'N/A', 'n/a', ''}
    
    try:
        worksheet = get_matched_distances_sheet()
        if not worksheet:
            return []
        
        all_data = worksheet.get_all_values()
        failed = []
        
        for i, row in enumerate(all_data[1:], start=2):  # Skip header, track row number
            # Need at least 5 columns (pickup_ar, dest_ar, pickup_en, dest_en)
            if len(row) >= 5:
                pickup_ar = row[1] if len(row) > 1 else ''
                dest_ar = row[2] if len(row) > 2 else ''
                pickup_en = row[3] if len(row) > 3 else ''
                dest_en = row[4] if len(row) > 4 else ''
                distance_value = row[6] if len(row) > 6 else ''  # Column G - Distance_Value
                status = row[7] if len(row) > 7 else ''  # Column H - Status
                added_to_pickle = row[8] if len(row) > 8 else 'No'  # Column I
                
                # Skip if already added to pickle
                if added_to_pickle == 'Yes':
                    continue
                
                # Skip if no city data
                if not pickup_ar or not dest_ar:
                    continue
                
                # Check if this is a failed lookup
                dist_str = str(distance_value).strip()
                is_failed = False
                error_type = ''
                
                # Check for known error values
                if dist_str in ERROR_VALUES or dist_str.startswith('#'):
                    is_failed = True
                    error_type = dist_str if dist_str else 'Empty'
                else:
                    # Try to parse as number - if it fails, it's an error
                    try:
                        dist_clean = dist_str.replace(',', '').replace(' km', '').replace('km', '').strip()
                        dist_km = float(dist_clean)
                        if dist_km <= 0:
                            is_failed = True
                            error_type = f'Invalid ({dist_km})'
                    except (ValueError, TypeError):
                        # Not a valid number - this is a failed lookup
                        if dist_str:  # Only count as failed if there's some value (not just empty/pending)
                            is_failed = True
                            error_type = dist_str[:20] + '...' if len(dist_str) > 20 else dist_str
                
                if is_failed:
                    failed.append({
                        'row': i,
                        'pickup_ar': pickup_ar,
                        'dest_ar': dest_ar,
                        'pickup_en': pickup_en,
                        'dest_en': dest_en,
                        'error_value': error_type,
                        'status': status
                    })
        
        return failed
    except Exception as e:
        return []

def mark_distances_as_added(row_numbers):
    """Mark rows in MatchedDistances sheet as added to pickle (Batch Update)."""
    try:
        import gspread
        worksheet = get_matched_distances_sheet()
        if not worksheet or not row_numbers:
            return False
        
        # Create a list of cell objects to update in one go
        # Column 9 is Column 'I'
        cells_to_update = [gspread.Cell(row=r, col=9, value='Yes') for r in row_numbers]
        
        # Send ALL updates in a single API call
        worksheet.update_cells(cells_to_update)
        
        return True
    except Exception as e:
        # Log the error to see if it's still failing
        print(f"Error marking rows: {e}")
        return False
        
def update_distance_pickle_from_sheet():
    """
    Pull resolved distances from MatchedDistances sheet and update the distance pickle.
    Returns (success, count_added, message)
    """
    global DISTANCE_MATRIX
    
    resolved = get_resolved_distances_from_sheet()
    if not resolved:
        return True, 0, "No new distances to add"
    
    # Load existing pickle
    pkl_path = os.path.join(APP_DIR, 'model_export', 'distance_matrix.pkl')
    
    try:
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                distance_matrix = pickle.load(f)
        else:
            distance_matrix = {}
        
        added_count = 0
        updated_count = 0
        skipped_count = 0
        rows_to_mark = []
        
        for item in resolved:
            key = (item['pickup_ar'], item['dest_ar'])
            rev_key = (item['dest_ar'], item['pickup_ar'])
            new_dist = item['distance_km']
            is_suggestion = item.get('is_suggestion', False)
            
            # Check if this is a new entry or an update
            if key in distance_matrix:
                old_dist = distance_matrix[key]
                if old_dist != new_dist:
                    # Only allow overwriting if it's a manual user suggestion
                    if is_suggestion:
                        distance_matrix[key] = new_dist
                        distance_matrix[rev_key] = new_dist  # Also update reverse
                        updated_count += 1
                        rows_to_mark.append(item['row'])
                    else:
                        # API result trying to overwrite - skip but don't mark as added
                        skipped_count += 1
                else:
                    # Same value, just mark as added
                    rows_to_mark.append(item['row'])
            else:
                # New entry - always add
                distance_matrix[key] = new_dist
                distance_matrix[rev_key] = new_dist
                added_count += 1
                rows_to_mark.append(item['row'])
        
        # Save updated pickle
        with open(pkl_path, 'wb') as f:
            pickle.dump(distance_matrix, f)
        
        # Update global variable
        DISTANCE_MATRIX = distance_matrix
        
        # Mark rows as added in sheet
        mark_distances_as_added(rows_to_mark)
        
        # Build message
        parts = []
        if added_count > 0:
            parts.append(f"{added_count} new")
        if updated_count > 0:
            parts.append(f"{updated_count} updated")
        if skipped_count > 0:
            parts.append(f"{skipped_count} skipped (already exists)")
        
        if parts:
            return True, added_count + updated_count, f"Distances: {', '.join(parts)}"
        else:
            return True, 0, "No changes needed (distances already match)"
    
    except Exception as e:
        return False, 0, f"Error updating pickle: {str(e)}"

# ============================================
# BULK ADJUSTMENTS SHEET - Log user-provided distances/coordinates
# These are for review only and NOT added to permanent references
# ============================================
def get_bulk_adjustments_sheet():
    """
    Get or create the Bulk Adjustments sheet for logging user-provided distances/coordinates.
    NOT cached because we need fresh connection for reads/writes.
    """
    try:
        client = get_gsheet_client()
        if client is None:
            return None
        
        sheet_url = st.secrets.get('error_log_sheet_url')
        if not sheet_url:
            return None
        
        spreadsheet = client.open_by_url(sheet_url)
        
        try:
            worksheet = spreadsheet.worksheet('Bulk Adjustments')
        except:
            # Create sheet with headers
            worksheet = spreadsheet.add_worksheet(title='Bulk Adjustments', rows=1000, cols=15)
            headers = [
                'Timestamp', 'Row_Number', 'Pickup_City', 'Destination_City', 
                'Pickup_EN', 'Destination_EN', 'User_Distance', 
                'Pickup_Lat', 'Pickup_Lon', 'Dropoff_Lat', 'Dropoff_Lon',
                'Detected_Pickup_Province', 'Detected_Dropoff_Province',
                'Detected_Pickup_Region', 'Detected_Dropoff_Region'
            ]
            worksheet.update('A1:O1', [headers])
        
        return worksheet
    except Exception as e:
        return None

def log_bulk_adjustment(row_num, pickup_city, dest_city, pickup_en, dest_en, 
                        user_distance=None, pickup_lat=None, pickup_lon=None, 
                        dropoff_lat=None, dropoff_lon=None,
                        pickup_province=None, dropoff_province=None,
                        pickup_region=None, dropoff_region=None):
    """
    Log a user-provided adjustment (distance or coordinates) to the pending queue.
    These are logged for review but NOT added to permanent references.
    
    Call flush_bulk_adjustments_to_sheet() at the end of bulk operations.
    """
    if 'bulk_adjustments_pending' not in st.session_state:
        st.session_state.bulk_adjustments_pending = []
    
    row = [
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        str(row_num),
        pickup_city,
        dest_city,
        pickup_en,
        dest_en,
        str(user_distance) if user_distance is not None else '',
        str(pickup_lat) if pickup_lat is not None else '',
        str(pickup_lon) if pickup_lon is not None else '',
        str(dropoff_lat) if dropoff_lat is not None else '',
        str(dropoff_lon) if dropoff_lon is not None else '',
        pickup_province or '',
        dropoff_province or '',
        pickup_region or '',
        dropoff_region or ''
    ]
    
    st.session_state.bulk_adjustments_pending.append(row)

def flush_bulk_adjustments_to_sheet(batch_size=500):
    """
    Flush pending bulk adjustments to Google Sheets in batches.
    Call this at the end of bulk operations.
    
    Args:
        batch_size: Max number of rows to write per API call (default 500)
    
    Returns: (success, count) tuple
    """
    if 'bulk_adjustments_pending' not in st.session_state or len(st.session_state.bulk_adjustments_pending) == 0:
        return True, 0
    
    pending = st.session_state.bulk_adjustments_pending
    total_count = len(pending)
    
    try:
        worksheet = get_bulk_adjustments_sheet()
        if worksheet:
            # Write in batches of batch_size
            written = 0
            for i in range(0, total_count, batch_size):
                batch = pending[i:i + batch_size]
                worksheet.append_rows(batch)
                written += len(batch)
            
            st.session_state.bulk_adjustments_pending = []
            return True, written
    except Exception as e:
        # Keep pending for retry
        return False, 0
    
    return True, 0

# ============================================
# BULK SHEET HELPER - NO CACHING (connections can go stale)
# ============================================
def get_bulk_sheet():
    """
    Get or create the All Lanes sheet.
    
    NOTE: This function is NOT cached because:
    1. Google Sheets connections can become stale/expire
    2. Each upload should get a fresh connection
    3. Caching the worksheet object leads to silent failures
    """
    try:
        client = get_gsheet_client()
        if client is None:
            return None
        
        spreadsheet = client.open_by_url(BULK_PRICING_SHEET_URL)
        
        try:
            worksheet = spreadsheet.worksheet(BULK_TAB_NAME)
        except:
            # Create the sheet if it doesn't exist (with enough rows for master grid)
            worksheet = spreadsheet.add_worksheet(title=BULK_TAB_NAME, rows=10000, cols=20)
            
        return worksheet
    except Exception as e:
        st.error(f"Google Sheets connection error: {str(e)}")
        return None

def ensure_sheet_rows(worksheet, required_rows):
    """
    Ensure the worksheet has enough rows. Expands if needed.
    
    Args:
        worksheet: gspread worksheet object
        required_rows: minimum number of rows needed
    """
    try:
        current_rows = worksheet.row_count
        if current_rows < required_rows:
            # Add some buffer (20% extra)
            new_rows = int(required_rows * 1.2)
            worksheet.resize(rows=new_rows)
            return True
    except Exception as e:
        pass
    return False

def log_exception(exception_type, details, immediate=False):
    """
    Log exception to session state (and optionally to Google Sheets immediately).
    
    By default, errors are queued locally and should be flushed at the end of 
    bulk operations using flush_error_log_to_sheet().
    
    Args:
        exception_type: Type of error (e.g., 'unmatched_city', 'missing_distance')
        details: Dict with error details
        immediate: If True, write to Google Sheet immediately (for single operations)
    """
    if 'error_log' not in st.session_state:
        st.session_state.error_log = []
    
    if 'error_log_pending' not in st.session_state:
        st.session_state.error_log_pending = []
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'type': exception_type,
        **details
    }
    st.session_state.error_log.append(log_entry)
    
    # Build the row for Google Sheets
    row = [
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        exception_type,
        details.get('pickup_city', details.get('original_value', '')),
        details.get('destination_city', details.get('normalized_to', '')),
        details.get('pickup_en', ''),
        details.get('destination_en', ''),
        str(details)
    ]
    
    if immediate:
        # Write immediately (for single route operations)
        try:
            worksheet = get_error_sheet()
            if worksheet:
                worksheet.append_row(row)
        except Exception as e:
            pass
    else:
        # Queue for batch write
        st.session_state.error_log_pending.append(row)

def flush_error_log_to_sheet(batch_size=500):
    """
    Flush pending error logs to Google Sheets in batches.
    Call this periodically during bulk operations to avoid accumulating too many.
    
    Args:
        batch_size: Max number of errors to write per API call (default 500)
    
    Returns: (success, count) tuple
    """
    if 'error_log_pending' not in st.session_state or len(st.session_state.error_log_pending) == 0:
        return True, 0
    
    pending = st.session_state.error_log_pending
    total_count = len(pending)
    
    try:
        worksheet = get_error_sheet()
        if worksheet:
            # Write in batches of batch_size
            written = 0
            for i in range(0, total_count, batch_size):
                batch = pending[i:i + batch_size]
                worksheet.append_rows(batch)
                written += len(batch)
            
            st.session_state.error_log_pending = []
            return True, written
    except Exception as e:
        # Keep pending for retry
        return False, 0
    
    return True, 0

def report_distance_issue(pickup_ar, dest_ar, pickup_en, dest_en, current_distance, issue_type, user_notes=''):
    """Report a distance issue to the Reported sheet."""
    try:
        worksheet = get_reported_sheet()
        if worksheet:
            row = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                pickup_ar,
                dest_ar,
                pickup_en,
                dest_en,
                str(current_distance),
                issue_type,
                user_notes
            ]
            worksheet.append_row(row)
            return True
    except Exception as e:
        pass
    return False

def suggest_distance_change(pickup_ar, dest_ar, pickup_en, dest_en, current_distance, suggested_distance, user_name):
    """
    Suggest a distance change for a lane.
    Logs to both Reported sheet (for tracking) and MatchedDistances sheet (for applying).
    
    Args:
        pickup_ar, dest_ar: Arabic city names
        pickup_en, dest_en: English city names
        current_distance: Current distance in km
        suggested_distance: User's suggested distance in km
        user_name: Name of user suggesting the change
    
    Returns: (success, message)
    """
    success_reported = False
    success_matched = False
    
    # 1. Log to Reported sheet with user info and distance change
    try:
        worksheet = get_reported_sheet()
        if worksheet:
            row = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                pickup_ar,
                dest_ar,
                pickup_en,
                dest_en,
                str(current_distance),
                'Distance suggestion',
                f"User: {user_name} | Suggested: {suggested_distance} km (was {current_distance} km)"
            ]
            worksheet.append_row(row)
            success_reported = True
    except Exception as e:
        pass
    
    # 2. Log to MatchedDistances sheet (ready to be applied)
    try:
        worksheet = get_matched_distances_sheet()
        if worksheet:
            # Check if this pair already exists
            existing = worksheet.get_all_values()
            row_to_update = None
            for i, row in enumerate(existing[1:], start=2):
                if len(row) >= 3:
                    if (row[1] == pickup_ar and row[2] == dest_ar) or \
                       (row[1] == dest_ar and row[2] == pickup_ar):
                        row_to_update = i
                        break
            
            # Convert suggested_distance to int for cleaner storage
            dist_int = int(round(suggested_distance))
            
            if row_to_update:
                # Update existing row - update all relevant columns in one call
                worksheet.update(f'F{row_to_update}:I{row_to_update}', [[
                    '',  # Column F - clear any formula
                    str(dist_int),  # Column G - distance value
                    f'Suggested by {user_name}',  # Column H - status
                    'No'  # Column I - not yet added to pickle
                ]])
            else:
                # Add new row
                next_row = len(existing) + 1
                row_data = [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    pickup_ar,
                    dest_ar,
                    pickup_en,
                    dest_en,
                    '',  # No formula - manual entry
                    str(dist_int),  # Direct value as string
                    f'Suggested by {user_name}',
                    'No'
                ]
                worksheet.update(f'A{next_row}:I{next_row}', [row_data])
            
            success_matched = True
    except Exception as e:
        pass
    
    if success_reported and success_matched:
        return True, "Distance suggestion logged and ready for import"
    elif success_reported:
        return True, "Logged to reports but failed to add to distance queue"
    elif success_matched:
        return True, "Added to distance queue but failed to log report"
    else:
        return False, "Failed to log suggestion"

def get_error_log_csv():
    """Get error log as CSV string."""
    if 'error_log' not in st.session_state or len(st.session_state.error_log) == 0:
        return None
    return pd.DataFrame(st.session_state.error_log).to_csv(index=False)

def clear_error_log():
    """Clear the session error log."""
    st.session_state.error_log = []

def upload_to_gsheet(df, batch_size=500, progress_callback=None):
    """
    Upload dataframe to Google Sheet in batches.
    
    Gets a FRESH connection each time (not cached) to avoid stale connection issues.
    Writes data in batches to avoid API limits and timeouts.
    
    Args:
        df: DataFrame to upload
        batch_size: Number of rows per batch (default 500)
        progress_callback: Optional function to call with progress updates
    """
    import time
    
    try:
        wks = get_bulk_sheet()
        if not wks:
            return False, "❌ Google Cloud credentials not found or sheet access denied. Check secrets configuration."
        
        # Clear existing data first
        try:
            wks.clear()
        except Exception as e:
            return False, f"❌ Failed to clear sheet: {str(e)}"
        
        # Convert all values to string for JSON serialization (gspread requirement)
        headers = df.columns.values.tolist()
        data_rows = df.astype(str).values.tolist()
        total_rows = len(data_rows)
        
        # Write header row first
        try:
            wks.update('A1', [headers])
        except Exception as e:
            return False, f"❌ Failed to write headers: {str(e)}"
        
        # Write data in batches
        rows_written = 0
        batch_num = 0
        
        for i in range(0, total_rows, batch_size):
            batch = data_rows[i:i + batch_size]
            start_row = i + 2  # +2 because row 1 is header, and sheets are 1-indexed
            
            try:
                # Calculate the range for this batch
                range_str = f'A{start_row}'
                
                wks.update(range_str, batch)
                rows_written += len(batch)
                batch_num += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(rows_written, total_rows, batch_num)
                
                # Small delay between batches to avoid rate limiting
                if i + batch_size < total_rows:
                    time.sleep(0.5)
                    
            except Exception as e:
                return False, f"❌ Failed at batch {batch_num + 1} (rows {start_row}-{start_row + len(batch) - 1}): {str(e)}"
        
        # Add timestamp row at the end
        timestamp_row = [f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
        try:
            wks.update(f'A{total_rows + 2}', [timestamp_row])
        except:
            pass  # Non-critical, ignore if it fails
            
        return True, f"✅ Successfully uploaded {rows_written} rows to '{BULK_TAB_NAME}' in {batch_num} batches"
    except Exception as e:
        return False, f"❌ Upload error: {str(e)}"

def upload_to_rfq_sheet(df, sheet_name, batch_size=500, progress_callback=None):
    """
    Upload dataframe to a NEW sheet tab in the RFQ Google Sheet.
    
    Creates a new sheet tab with the given name to prevent overwriting existing data.
    
    Args:
        df: DataFrame to upload
        sheet_name: Name for the new sheet tab (must be unique)
        batch_size: Number of rows per batch (default 500)
        progress_callback: Optional function to call with progress updates
    
    Returns:
        (success, message) tuple
    """
    import time
    
    try:
        client = get_gsheet_client()
        if client is None:
            return False, "❌ Google Cloud credentials not found. Check secrets configuration."
        
        rfq_url = st.secrets.get('RFQ_url')
        if not rfq_url:
            return False, "❌ RFQ_url not configured in secrets."
        
        try:
            spreadsheet = client.open_by_url(rfq_url)
        except Exception as e:
            return False, f"❌ Failed to open RFQ spreadsheet: {str(e)}"
        
        # Check if sheet name already exists
        existing_sheets = [ws.title for ws in spreadsheet.worksheets()]
        if sheet_name in existing_sheets:
            return False, f"❌ Sheet '{sheet_name}' already exists. Please choose a different name."
        
        # Create new sheet tab
        try:
            total_rows = len(df) + 5  # Data rows + header + buffer
            total_cols = len(df.columns) + 2  # Data cols + buffer
            wks = spreadsheet.add_worksheet(title=sheet_name, rows=total_rows, cols=total_cols)
        except Exception as e:
            return False, f"❌ Failed to create sheet '{sheet_name}': {str(e)}"
        
        # Convert all values to string for JSON serialization (gspread requirement)
        headers = df.columns.values.tolist()
        data_rows = df.astype(str).values.tolist()
        total_data_rows = len(data_rows)
        
        # Write header row first
        try:
            wks.update('A1', [headers])
        except Exception as e:
            return False, f"❌ Failed to write headers: {str(e)}"
        
        # Write data in batches
        rows_written = 0
        batch_num = 0
        
        for i in range(0, total_data_rows, batch_size):
            batch = data_rows[i:i + batch_size]
            start_row = i + 2  # +2 because row 1 is header, and sheets are 1-indexed
            
            try:
                range_str = f'A{start_row}'
                wks.update(range_str, batch)
                rows_written += len(batch)
                batch_num += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(rows_written, total_data_rows, batch_num)
                
                # Small delay between batches to avoid rate limiting
                if i + batch_size < total_data_rows:
                    time.sleep(0.5)
                    
            except Exception as e:
                return False, f"❌ Failed at batch {batch_num + 1} (rows {start_row}-{start_row + len(batch) - 1}): {str(e)}"
        
        # Add timestamp row at the end
        timestamp_row = [f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
        try:
            wks.update(f'A{total_data_rows + 2}', [timestamp_row])
        except:
            pass  # Non-critical, ignore if it fails
            
        return True, f"✅ Successfully uploaded {rows_written} rows to new sheet '{sheet_name}' in {batch_num} batches"
    except Exception as e:
        return False, f"❌ Upload error: {str(e)}"

def populate_rfq_template_sheet(template_df):
    """
    Populate the 'Template' sheet in the RFQ spreadsheet with template data.
    
    Creates the sheet if it doesn't exist, or clears and repopulates if it does.
    
    Args:
        template_df: DataFrame containing the template structure
    
    Returns:
        (success, message) tuple
    """
    try:
        client = get_gsheet_client()
        if client is None:
            return False, "❌ Google Cloud credentials not found. Check secrets configuration."
        
        rfq_url = st.secrets.get('RFQ_url')
        if not rfq_url:
            return False, "❌ RFQ_url not configured in secrets."
        
        try:
            spreadsheet = client.open_by_url(rfq_url)
        except Exception as e:
            return False, f"❌ Failed to open RFQ spreadsheet: {str(e)}"
        
        # Check if Template sheet exists
        existing_sheets = [ws.title for ws in spreadsheet.worksheets()]
        
        if 'Template' in existing_sheets:
            # Get existing sheet and clear it
            wks = spreadsheet.worksheet('Template')
            wks.clear()
        else:
            # Create new Template sheet
            wks = spreadsheet.add_worksheet(title='Template', rows=100, cols=10)
        
        # Write headers and data
        headers = template_df.columns.values.tolist()
        data_rows = template_df.astype(str).values.tolist()
        
        # Replace 'nan' with empty string
        data_rows = [['' if cell == 'nan' else cell for cell in row] for row in data_rows]
        
        # Write all data at once (header + data)
        all_data = [headers] + data_rows
        wks.update('A1', all_data)
        
        return True, "✅ Template sheet updated successfully!"
    except Exception as e:
        return False, f"❌ Error updating template: {str(e)}"

# ============================================
# FIXED SETTINGS
# ============================================
# Sampling parameters for ammunition display
N_SIMILAR = 10          # Number of similar loads to consider
MIN_DAYS_APART = 5      # Minimum days between samples (prevents clustering)
MAX_AGE_DAYS = 180      # Maximum age of loads to show as ammunition
RECENT_PRIORITY_DAYS = 30  # Prioritize loads within this window

# Recency model window - loads within this window use recency-based pricing
RECENCY_WINDOW = 90     # Days to consider for recency model

# Rental cost index parameters (reference for minimum viable price)
RENTAL_COST_PER_DAY = 800   # SAR per day rental cost
RENTAL_KM_PER_DAY = 600     # km that equals one day of rental

# Error bar percentages by confidence level (for price ranges)
ERROR_BARS = {'High': 0.10, 'Medium': 0.15, 'Low': 0.25, 'Very Low': 0.35}

# Margin settings based on backhaul probability
# Higher margin when backhaul is unlikely (carrier has to deadhead back)
MARGIN_HIGH_BACKHAUL = 0.12      # 12% margin when good backhaul probability
MARGIN_MEDIUM_BACKHAUL = 0.18    # 18% margin for medium backhaul
MARGIN_LOW_BACKHAUL = 0.25       # 25% margin when poor backhaul (deadhead likely)
MARGIN_UNKNOWN = 0.20            # 20% default when no backhaul data available

# Backhaul thresholds (inbound/outbound CPK ratio)
# High ratio = good inbound demand = carrier likely finds return load
BACKHAUL_HIGH_THRESHOLD = 0.85   # Ratio > 0.85 = High backhaul probability
BACKHAUL_MEDIUM_THRESHOLD = 0.65 # Ratio 0.65-0.85 = Medium backhaul probability
# Below 0.65 = Low backhaul probability (carrier likely deadheads)

# ============================================
# LOAD CITY NORMALIZATION
# Maps city name variants (Arabic, English, misspellings) to canonical forms
# Also loads region mappings for regional pricing models
# ============================================
@st.cache_resource
def load_city_normalization():
    """
    Load city normalization from CSV.
    
    The CSV contains:
    - variant: Different ways a city might be written (Arabic, English, typos)
    - canonical: The standard Arabic form we normalize to
    - region: Geographic region (5 macro-regions: Eastern, Western, Central, Northern, Southern)
    - province: Official Saudi province (13 provinces, e.g., "Riyadh Province", "Eastern Province")
    
    Returns multiple lookup dictionaries for flexible matching.
    """
    norm_path = os.path.join(APP_DIR, 'model_export', 'city_normalization_with_regions.csv')
    
    variant_to_canonical = {}           # Exact match lookup
    variant_to_canonical_lower = {}     # Lowercase lookup for English names
    variant_to_canonical_aggressive = {} # Aggressive normalization (no spaces/punctuation)
    variant_to_region = {}              # Variant -> Region mapping (5 regions)
    canonical_to_region = {}            # Canonical -> Region mapping (5 regions)
    variant_to_province = {}            # Variant -> Province mapping (13 provinces)
    canonical_to_province = {}          # Canonical -> Province mapping (13 provinces)
    canonical_to_english = {}           # Canonical Arabic -> English display name
    
    if os.path.exists(norm_path):
        try:
            df = pd.read_csv(norm_path)
            
            for _, row in df.iterrows():
                variant = str(row['variant']).strip()
                canonical = str(row['canonical']).strip()
                region = row['region'] if pd.notna(row.get('region')) else None
                province = row['province'] if pd.notna(row.get('province')) else None

                variant_to_canonical[variant] = canonical
                
                # Also add lowercase version for case-insensitive English matching
                variant_lower = normalize_english_text(variant)
                if variant_lower:
                    variant_to_canonical_lower[variant_lower] = canonical
                
                # Add aggressive normalized version (strips all spaces/punctuation)
                variant_aggressive = normalize_aggressive(variant)
                if variant_aggressive:
                    variant_to_canonical_aggressive[variant_aggressive] = canonical
                
                # Region mappings (5 regions - for Index+Shrinkage fallback)
                if region:
                    variant_to_region[variant] = region
                    if canonical not in canonical_to_region:
                        canonical_to_region[canonical] = region
                
                # Province mappings (13 provinces - for Blend model)
                if province:
                    variant_to_province[variant] = province
                    if canonical not in canonical_to_province:
                        canonical_to_province[canonical] = province
            
            # Build English display names from variants
            # Prefer title-case ASCII variants as English names
            grouped = df.groupby('canonical')['variant'].apply(list)
            for canonical, variants in grouped.items():
                english_name = None
                ascii_variants = [v for v in variants if re.match(r'^[A-Za-z\s\-\(\)\.]+$', str(v))]
                
                if ascii_variants:
                    title_case = [v for v in ascii_variants if str(v)[0].isupper() and not str(v).isupper()]
                    english_name = title_case[0] if title_case else ascii_variants[0]
                
                canonical_to_english[canonical] = english_name if english_name else canonical

        except Exception as e:
            st.error(f"Error reading normalization file: {e}")
            
    return (variant_to_canonical, variant_to_canonical_lower, variant_to_canonical_aggressive,
            variant_to_region, canonical_to_region, variant_to_province, canonical_to_province,
            canonical_to_english)

# Load mappings
(CITY_TO_CANONICAL, CITY_TO_CANONICAL_LOWER, CITY_TO_CANONICAL_AGGRESSIVE,
 CITY_TO_REGION, CANONICAL_TO_REGION, CITY_TO_PROVINCE, CANONICAL_TO_PROVINCE,
 CITY_AR_TO_EN) = load_city_normalization()
CITY_EN_TO_AR = {v: k for k, v in CITY_AR_TO_EN.items()}

# ============================================
# HELPER FUNCTIONS
# City normalization and translation utilities
# ============================================

# OSM SEARCH HELPERS
def search_osm(query):
    """Search OpenStreetMap via Nominatim API."""
    if not query: return None
    url = "https://nominatim.openstreetmap.org/search"
    params = {'q': query, 'format': 'json', 'limit': 1, 'countrycodes': 'sa'}
    headers = {'User-Agent': 'FreightPricingTool/1.0'}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon']), data[0]['display_name']
    except: return None
    return None

def decode_polyline(polyline_str):
    """Decodes a Polyline string into a list of lat/lon pairs."""
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'latitude': 0, 'longitude': 0}
    length = len(polyline_str)

    while index < length:
        for unit in ['latitude', 'longitude']:
            shift, result = 0, 0
            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if not byte >= 0x20: break
            if (result & 1): changes[unit] = ~(result >> 1)
            else: changes[unit] = (result >> 1)
        lat += changes['latitude']
        lng += changes['longitude']
        coordinates.append((lat / 100000.0, lng / 100000.0))
    return coordinates

def get_osrm_route(start_coords, end_coords):
    """
    Calculate driving distance AND route geometry.
    Returns (distance_km, route_coordinates_list) or (None, None).
    """
    # Request 'overview=full' to get the detailed road geometry
    url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
    params = {'overview': 'full', 'geometries': 'polyline'}
    
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data.get('code') == 'Ok' and data.get('routes'):
                route = data['routes'][0]
                dist_km = route['distance'] / 1000.0
                # Decode the encoded geometry string into points
                geometry = decode_polyline(route['geometry'])
                return dist_km, geometry
    except: return None, None
    return None, None
    
def normalize_city(city_raw):
    """
    Normalize city name to standard Arabic canonical form.
    
    Lookup order:
    1. Exact match in normalization CSV
    2. Lowercase/normalized English match
    3. Aggressive match (strips all spaces/punctuation)
    4. Reverse lookup from English->Arabic mapping
    
    Returns: (canonical_name, was_matched)
    """
    if pd.isna(city_raw) or city_raw == '':
        return None, False
    city = str(city_raw).strip()
    
    # 1. Try exact match first
    if city in CITY_TO_CANONICAL:
        return CITY_TO_CANONICAL[city], True
    
    # 2. Try lowercase/normalized lookup (handles "JEDDAH", "jeddah", etc.)
    city_lower = normalize_english_text(city)
    if city_lower and city_lower in CITY_TO_CANONICAL_LOWER:
        return CITY_TO_CANONICAL_LOWER[city_lower], True
    
    # 3. Try aggressive normalization (handles "Al-Jubail" vs "AlJubail" vs "Al Jubail")
    city_aggressive = normalize_aggressive(city)
    if city_aggressive and city_aggressive in CITY_TO_CANONICAL_AGGRESSIVE:
        return CITY_TO_CANONICAL_AGGRESSIVE[city_aggressive], True
    
    # 4. Try reverse lookup from English display names
    if city in CITY_EN_TO_AR:
        return CITY_EN_TO_AR[city], True

    # Return as-is with match=False (will be logged as unmatched)
    return city, False

# ============================================
# FUZZY MATCHING FOR CITY RESOLUTION
# Uses rapidfuzz for finding best matches to unmatched city names
# ============================================
def get_all_variants_for_fuzzy_matching():
    """
    Build a list of all variants with their canonical mappings for fuzzy matching.
    Prioritizes English variants over Arabic for better bulk input matching.
    
    Returns: List of (variant, canonical, is_english) tuples sorted by priority
    """
    variants = []
    
    for variant, canonical in CITY_TO_CANONICAL.items():
        # Check if variant is primarily English (ASCII letters)
        is_english = bool(re.match(r'^[A-Za-z\s\-\(\)\.\']+$', str(variant)))
        variants.append((variant, canonical, is_english))
    
    # Sort: English variants first, then Arabic
    variants.sort(key=lambda x: (not x[2], x[0]))
    return variants

# Cache the variants list
FUZZY_VARIANTS_LIST = get_all_variants_for_fuzzy_matching() if RAPIDFUZZ_AVAILABLE else []

def fuzzy_match_city(city_raw, threshold=80):
    """
    Find the best fuzzy match for an unmatched city name.
    
    Args:
        city_raw: The unmatched city name to find a match for
        threshold: Minimum similarity score (0-100) to consider a match
    
    Returns:
        dict with:
            - match_found: bool
            - suggested_canonical: The canonical name of the best match (if found)
            - suggested_variant: The variant that matched (if found)
            - confidence: Similarity score (0-100)
            - all_matches: List of top matches above threshold
    """
    if not RAPIDFUZZ_AVAILABLE:
        return {
            'match_found': False,
            'error': 'rapidfuzz not installed'
        }
    
    if pd.isna(city_raw) or str(city_raw).strip() == '':
        return {
            'match_found': False,
            'error': 'Empty input'
        }
    
    city = str(city_raw).strip()
    
    # Extract just the variants for matching
    variant_list = [v[0] for v in FUZZY_VARIANTS_LIST]
    
    # Use rapidfuzz to find best matches
    # We use token_sort_ratio for better handling of word order differences
    matches = process.extract(
        city, 
        variant_list, 
        scorer=fuzz.token_sort_ratio,
        limit=5
    )
    
    # Filter matches above threshold
    good_matches = [(match, score, idx) for match, score, idx in matches if score >= threshold]
    
    if not good_matches:
        return {
            'match_found': False,
            'input': city,
            'best_score': matches[0][1] if matches else 0,
            'best_variant': matches[0][0] if matches else None
        }
    
    # Get the best match
    best_variant, best_score, best_idx = good_matches[0]
    best_canonical = FUZZY_VARIANTS_LIST[best_idx][1]
    
    # Build list of all good matches with their canonicals
    all_matches = []
    for match_variant, score, idx in good_matches:
        canonical = FUZZY_VARIANTS_LIST[idx][1]
        all_matches.append({
            'variant': match_variant,
            'canonical': canonical,
            'score': score
        })
    
    return {
        'match_found': True,
        'input': city,
        'suggested_canonical': best_canonical,
        'suggested_variant': best_variant,
        'confidence': best_score,
        'is_high_confidence': best_score >= 90,
        'all_matches': all_matches
    }

def batch_fuzzy_match_cities(unmatched_cities, threshold=80):
    """
    Find fuzzy matches for a list of unique unmatched city names.
    
    Args:
        unmatched_cities: List of unique city names that didn't match
        threshold: Minimum similarity score to suggest a match
    
    Returns:
        Dict mapping each input city to its fuzzy match result
    """
    results = {}
    for city in unmatched_cities:
        results[city] = fuzzy_match_city(city, threshold)
    return results

def get_city_region(city):
    """Get region for a city (checks both variant and canonical mappings). Returns 5-region value."""
    if city in CITY_TO_REGION:
        return CITY_TO_REGION[city]
    if city in CANONICAL_TO_REGION:
        return CANONICAL_TO_REGION[city]
    # Try normalizing first then looking up
    canonical = CITY_TO_CANONICAL.get(city, city)
    if canonical in CANONICAL_TO_REGION:
        return CANONICAL_TO_REGION[canonical]
    return None

def get_city_province(city):
    """Get province for a city (checks both variant and canonical mappings). Returns 13-province value."""
    if city in CITY_TO_PROVINCE:
        return CITY_TO_PROVINCE[city]
    if city in CANONICAL_TO_PROVINCE:
        return CANONICAL_TO_PROVINCE[city]
    # Try normalizing first then looking up
    canonical = CITY_TO_CANONICAL.get(city, city)
    if canonical in CANONICAL_TO_PROVINCE:
        return CANONICAL_TO_PROVINCE[canonical]
    return None

def classify_unmatched_city(city_raw):
    """
    Classify why a city didn't match and determine the error type.
    
    Error Types:
    1. 'no_region' - City exists in historical data (VALID_CITIES_AR) but not in CSV
                    → Needs to be added to canonicals CSV with region
    2. 'unknown_city' - City not in historical data AND not in CSV
                       → Possibly a new city or typo/variant of existing city
    3. 'empty' - Empty/null city name
    
    Returns dict with:
        - error_type: 'no_region' | 'unknown_city' | 'empty'
        - message: Human-readable explanation
        - action: Suggested action to take
        - in_historical_data: bool
    """
    if pd.isna(city_raw) or str(city_raw).strip() == '':
        return {
            'error_type': 'empty',
            'message': 'Empty city name',
            'action': 'Provide a valid city name',
            'in_historical_data': False
        }
    
    city = str(city_raw).strip()
    
    # Check if city exists in historical data (VALID_CITIES_AR)
    in_historical = city in VALID_CITIES_AR
    
    # Check if city has a region defined
    region = get_city_region(city)
    
    if in_historical and not region:
        # City is in historical data but not in normalization CSV
        return {
            'error_type': 'no_region',
            'message': f'City "{city}" exists in historical data but has no region defined',
            'action': 'Add this city to the canonicals CSV with its region',
            'in_historical_data': True
        }
    elif not in_historical:
        # City not in historical data - unknown/new city or possible variant
        return {
            'error_type': 'unknown_city',
            'message': f'City "{city}" not found in database',
            'action': 'Check if this is a variant of an existing city, or add as new city to CSV',
            'in_historical_data': False
        }
    
    # Shouldn't reach here if normalize_city returned False, but just in case
    return {
        'error_type': 'unknown',
        'message': f'City "{city}" - unknown issue',
        'action': 'Investigate manually',
        'in_historical_data': in_historical
    }

def to_english_city(city_ar):
    """Convert Arabic city name to English display name."""
    if city_ar in CITY_AR_TO_EN:
        return CITY_AR_TO_EN[city_ar]
    # Try normalizing first
    norm, found = normalize_city(city_ar)
    if found and norm in CITY_AR_TO_EN:
        return CITY_AR_TO_EN[norm]
    return city_ar

def to_arabic_city(city_en):
    """Convert English city name to canonical Arabic form."""
    if city_en in CITY_EN_TO_AR:
        return CITY_EN_TO_AR[city_en]
    norm, found = normalize_city(city_en)
    return norm if norm else city_en

# ============================================
# VEHICLE TYPE MAPPINGS
# ============================================
VEHICLE_TYPE_EN = {
    'تريلا فرش': 'Flatbed Trailer',
    'تريلا ستائر': 'Curtain Trailer',
    'تريلا مقفول': 'Closed Trailer',
    'تريلا ثلاجة': 'Refrigerated Trailer',
    'سطحة': 'Lowbed Trailer',     # Added Arabic for Lowbed
    'Lowbed Trailer': 'Lowbed Trailer',
    'دينا': 'Dynna',              # New
    'لوري': 'Lorries',            # New
    'Unknown': 'Unknown',
}
VEHICLE_TYPE_AR = {v: k for k, v in VEHICLE_TYPE_EN.items()}

DEFAULT_VEHICLE_AR = 'تريلا فرش'
DEFAULT_VEHICLE_EN = 'Flatbed Trailer'

def to_english_vehicle(vtype_ar):
    return VEHICLE_TYPE_EN.get(vtype_ar, vtype_ar)

def to_arabic_vehicle(vtype_en):
    if vtype_en in VEHICLE_TYPE_AR:
        return VEHICLE_TYPE_AR[vtype_en]
    if vtype_en in VEHICLE_TYPE_EN:
        return vtype_en
    return DEFAULT_VEHICLE_AR

# ============================================
# COMMODITY TRANSLATIONS
# ============================================
COMMODITY_EN = {
    'Unknown': 'Unknown', 'أكسيد الحديد': 'Iron Oxide', 'أكياس ورقية فارغة': 'Empty Paper Bags', 'أنابيب': 'Pipes',
    'اجهزه كهربائيه': 'Electrical Equipment', 'ارز': 'Rice', 'اسمدة': 'Fertilizer', 'الطرود': 'Parcels',
    'براميل': 'Barrels', 'بلاستيك': 'Plastic', 'تمر': 'Dates', 'جبس': 'Gypsum', 'خردة': 'Scrap', 'خشب': 'Wood',
    'رمل': 'Sand', 'رمل السيليكا': 'Silica Sand', 'زجاج ليفي': 'Fiberglass', 'زيوت': 'Oils', 'سكر': 'Sugar',
    'سلع إستهلاكية': 'Consumer Goods', 'سيراميك': 'Ceramics', 'فحم': 'Coal', 'فوارغ': 'Empties', 'كابلات': 'Cables',
    'كيماوي': 'Chemicals', 'لفات حديد': 'Steel Coils', 'معدات': 'Equipment', 'منتجات الصلب': 'Steel Products',
    'مواد بناء': 'Building Materials', 'مياه': 'Water', 'ورق': 'Paper',
}
COMMODITY_AR = {v: k for k, v in COMMODITY_EN.items()}

def to_english_commodity(commodity_ar):
    return COMMODITY_EN.get(commodity_ar, commodity_ar)

def to_arabic_commodity(commodity_en):
    return COMMODITY_AR.get(commodity_en, commodity_en)

# ============================================
# INDEX + SHRINKAGE MODEL (Rare Lane Model)
# Used for lanes WITH some historical data
# Combines market index trends with shrinkage estimation
# ============================================
class IndexShrinkagePredictor:
    """
    Index + Shrinkage model for lanes with some historical data.
    
    Two-component prediction:
    1. Index: Uses lane-specific multiplier × current market index
       - Captures how this lane moves relative to market
    2. Shrinkage: Bayesian estimate that shrinks lane mean toward city prior
       - Handles sparse data by borrowing strength from similar lanes
    
    Final prediction = average of Index and Shrinkage components
    """
    
    def __init__(self, model_artifacts):
        m = model_artifacts
        self.current_index = m['current_index']      # Current market index value
        self.lane_multipliers = m['lane_multipliers'] # Lane-specific multipliers
        self.global_mean = m['global_mean']          # Global mean CPK (fallback)
        self.k = m['k_prior_strength']               # Shrinkage strength parameter
        self.pickup_priors = m['pickup_priors']      # City-level priors for pickup
        self.dest_priors = m['dest_priors']          # City-level priors for destination
        self.lane_stats = m['lane_stats']            # Historical lane statistics
        self.city_to_region = m.get('city_to_region', {})
        self.regional_cpk = {tuple(k.split('|')): v for k, v in m.get('regional_cpk', {}).items()} if isinstance(list(m.get('regional_cpk', {}).keys())[0] if m.get('regional_cpk') else '', str) else m.get('regional_cpk', {})
        self.model_date = m.get('model_date', 'Unknown')
    
    def get_canonical_lane(self, lane):
        parts = lane.split(' → ')
        if len(parts) == 2:
            pickup = parts[0]
            dest = parts[1]
            p_can = CITY_TO_CANONICAL.get(pickup) or CITY_TO_CANONICAL_LOWER.get(normalize_english_text(pickup), pickup)
            d_can = CITY_TO_CANONICAL.get(dest) or CITY_TO_CANONICAL_LOWER.get(normalize_english_text(dest), dest)
            return f"{p_can} → {d_can}"
        return lane
    
    def has_lane_data(self, lane):
        if lane in self.lane_stats or lane in self.lane_multipliers:
            return True
        canonical_lane = self.get_canonical_lane(lane)
        return canonical_lane in self.lane_stats or canonical_lane in self.lane_multipliers
    
    def predict(self, pickup_city, dest_city, distance_km=None):
        lane = f"{pickup_city} → {dest_city}"
        
        # Try canonical lane if original doesn't exist
        canonical_lane = self.get_canonical_lane(lane)
        lookup_lane = canonical_lane if (canonical_lane in self.lane_stats or canonical_lane in self.lane_multipliers) else lane
        
        # Predict Index
        idx_pred = None
        if lookup_lane in self.lane_multipliers:
            idx_pred = self.current_index * self.lane_multipliers[lookup_lane]
        
        # Predict Shrinkage
        p_prior = self.pickup_priors.get(pickup_city)
        if p_prior is None:
            pickup_canonical = CITY_TO_CANONICAL.get(pickup_city, pickup_city)
            p_prior = self.pickup_priors.get(pickup_canonical, self.global_mean)
        
        d_prior = self.dest_priors.get(dest_city)
        if d_prior is None:
            dest_canonical = CITY_TO_CANONICAL.get(dest_city, dest_city)
            d_prior = self.dest_priors.get(dest_canonical, self.global_mean)
        
        city_prior = (p_prior + d_prior) / 2
        
        if lane in self.lane_stats:
            stats = self.lane_stats[lane]
        else:
            stats = self.lane_stats.get(lookup_lane)
            
        if stats:
            lane_mean = stats['lane_mean']
            lane_n = stats['lane_n']
            lam = lane_n / (lane_n + self.k)
            shrink_pred = lam * lane_mean + (1 - lam) * city_prior
        else:
            shrink_pred = city_prior
            
        # Combine
        if idx_pred is not None:
            predicted_cpk = (idx_pred + shrink_pred) / 2
            method = 'Index + Shrinkage'
            if stats:
                n = stats['lane_n']
                confidence = 'High' if n >= 20 else 'Medium' if n >= 5 else 'Low'
            else:
                confidence = 'Low'
        else:
            predicted_cpk = shrink_pred
            method = 'Shrinkage'
            confidence = 'Medium'
            
        error_pct = ERROR_BARS.get(confidence, 0.25)
        price_low = predicted_cpk * (1 - error_pct)
        price_high = predicted_cpk * (1 + error_pct)
        
        result = {
            'predicted_cpk': round(predicted_cpk, 3),
            'cpk_low': round(price_low, 3),
            'cpk_high': round(price_high, 3),
            'method': method,
            'confidence': confidence,
        }
        
        if distance_km and distance_km > 0:
            result['predicted_cost'] = round(predicted_cpk * distance_km, 0)
            result['cost_low'] = round(price_low * distance_km, 0)
            result['cost_high'] = round(price_high * distance_km, 0)
        
        return result

# ============================================
# BLEND MODEL (New Lane Model - 0.7 Province)
# Used for completely NEW lanes with NO historical data
# Blends province-level averages with city-level multipliers
# Uses 13 official Saudi provinces for more granular pricing
# ============================================
class BlendPredictor:
    """
    Blend model (0.7 Province + 0.3 City) for completely new lanes.
    
    For lanes with zero historical data, we estimate price by:
    1. Province CPK: Average CPK for the province pair (13 Saudi provinces)
       e.g., "Riyadh Province" → "Eastern Province"
    2. City Multipliers: Pickup and destination city adjustment factors
    
    Final: 70% Province + 30% City-adjusted
    
    This provides a reasonable estimate even for never-seen routes by
    leveraging geographic patterns in freight pricing at the province level.
    
    Note: Falls back to 5-region model if province data not available.
    """
    
    def __init__(self, model_artifacts):
        m = model_artifacts
        self.config = m['config']
        self.province_weight = self.config.get('province_weight', self.config.get('regional_weight', 0.7))  # 70% province
        self.current_index = m['current_index']
        self.pickup_city_mult = m['pickup_city_mult']   # City-specific adjustments
        self.dest_city_mult = m['dest_city_mult']
        
        # Province mappings (13 provinces) - primary for Blend model
        self.city_to_province = m.get('city_to_province', {})
        self.province_cpk = m.get('province_cpk', {})
        
        # Region mappings (5 regions) - fallback for backward compatibility
        self.city_to_region = m.get('city_to_region', {})
        self.regional_cpk = m.get('regional_cpk', {})
        
        self.model_date = self.config.get('training_date', 'Unknown')
        
        # Log model info for debugging
        n_provinces = len(set(self.city_to_province.values())) if self.city_to_province else 0
        n_regions = len(set(self.city_to_region.values())) if self.city_to_region else 0
        self.using_provinces = n_provinces > 5  # True if we have 13 provinces, not 5 regions
    
    def predict(self, pickup_city, dest_city, distance_km=None, 
                pickup_region_override=None, dest_region_override=None):
        """
        Predict CPK for a lane using province/region blending.
        
        Args:
            pickup_city: Pickup city (Arabic canonical or English)
            dest_city: Destination city (Arabic canonical or English)
            distance_km: Distance in km (optional, used to calculate total cost)
            pickup_region_override: Override region for pickup (from coordinates)
            dest_region_override: Override region for destination (from coordinates)
        """
        # Get canonical names
        p_can = CITY_TO_CANONICAL.get(pickup_city, pickup_city)
        d_can = CITY_TO_CANONICAL.get(dest_city, dest_city)
        
        # Try PROVINCE lookup first (13 provinces - more granular)
        p_province = CANONICAL_TO_PROVINCE.get(p_can) or self.city_to_province.get(p_can) or self.city_to_province.get(pickup_city)
        d_province = CANONICAL_TO_PROVINCE.get(d_can) or self.city_to_province.get(d_can) or self.city_to_province.get(dest_city)
        
        province_cpk = None
        if p_province and d_province:
            province_cpk = self.province_cpk.get((p_province, d_province))
        
        # If no province CPK found, try REGION fallback (5 regions)
        regional_cpk = None
        if province_cpk is None:
            # First try to get region from city mapping
            p_region = CANONICAL_TO_REGION.get(p_can) or self.city_to_region.get(p_can) or self.city_to_region.get(pickup_city)
            d_region = CANONICAL_TO_REGION.get(d_can) or self.city_to_region.get(d_can) or self.city_to_region.get(dest_city)
            
            # If still no region found, use coordinate-based region overrides
            if not p_region and pickup_region_override:
                p_region = pickup_region_override
            if not d_region and dest_region_override:
                d_region = dest_region_override
            
            if p_region and d_region:
                regional_cpk = self.regional_cpk.get((p_region, d_region))
        
        # City Multipliers (30% weight)
        p_mult = self.pickup_city_mult.get(pickup_city, self.pickup_city_mult.get(p_can, 1.0))
        d_mult = self.dest_city_mult.get(dest_city, self.dest_city_mult.get(d_can, 1.0))
        city_cpk = p_mult * d_mult * self.current_index
        
        # Determine which geographic CPK to use
        if province_cpk is not None:
            # Use province-level (primary - 13 provinces)
            predicted_cpk = self.province_weight * province_cpk + (1 - self.province_weight) * city_cpk
            method = f'Blend ({self.province_weight:.0%} Province)'
            confidence = 'Low'
        elif regional_cpk is not None:
            # Fallback to region-level (5 regions)
            predicted_cpk = self.province_weight * regional_cpk + (1 - self.province_weight) * city_cpk
            method = f'Blend ({self.province_weight:.0%} Regional)'
            # If we used coordinate-based regions, note it
            if pickup_region_override or dest_region_override:
                method = f'Blend ({self.province_weight:.0%} Regional/Coords)'
            confidence = 'Low'
        else:
            # No geographic data - use city multipliers only
            predicted_cpk = city_cpk
            method = 'City Multipliers'
            confidence = 'Very Low'
            
        error_pct = ERROR_BARS.get(confidence, 0.25)
        price_low = predicted_cpk * (1 - error_pct)
        price_high = predicted_cpk * (1 + error_pct)
        
        result = {
            'predicted_cpk': round(predicted_cpk, 3),
            'cpk_low': round(price_low, 3),
            'cpk_high': round(price_high, 3),
            'method': method,
            'confidence': confidence,
        }
        
        if distance_km and distance_km > 0:
            result['predicted_cost'] = round(predicted_cpk * distance_km, 0)
            result['cost_low'] = round(price_low * distance_km, 0)
            result['cost_high'] = round(price_high * distance_km, 0)
        
        return result

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    MODEL_DIR = os.path.join(APP_DIR, 'model_export')
    
    carrier_model = None
    json_path = os.path.join(MODEL_DIR, 'carrier_model.json')
    cbm_path = os.path.join(MODEL_DIR, 'carrier_model.cbm')
    if os.path.exists(json_path):
        carrier_model = CatBoostRegressor()
        carrier_model.load_model(json_path, format='json')
    elif os.path.exists(cbm_path):
        carrier_model = CatBoostRegressor()
        carrier_model.load_model(cbm_path)
    
    with open(os.path.join(MODEL_DIR, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)
    
    csv_path = os.path.join(MODEL_DIR, 'reference_data.csv')
    parquet_path = os.path.join(MODEL_DIR, 'reference_data.parquet')
    if os.path.exists(csv_path):
        df_knn = pd.read_csv(csv_path)
    else:
        df_knn = pd.read_parquet(parquet_path)
    
    # Load distance matrix
    distance_matrix_path = os.path.join(MODEL_DIR, 'distance_matrix.pkl')
    if os.path.exists(distance_matrix_path):
        with open(distance_matrix_path, 'rb') as f:
            distance_matrix = pickle.load(f)
    else:
        distance_matrix = {}
    
    # Load Index + Shrinkage model
    rare_lane_path = os.path.join(MODEL_DIR, 'rare_lane_models.pkl')
    index_shrink_predictor = None
    if os.path.exists(rare_lane_path):
        with open(rare_lane_path, 'rb') as f:
            rare_lane_artifacts = pickle.load(f)
        index_shrink_predictor = IndexShrinkagePredictor(rare_lane_artifacts)
    
    # Load Blend model
    blend_path = os.path.join(MODEL_DIR, 'new_lane_model_blend.pkl')
    blend_predictor = None
    if os.path.exists(blend_path):
        with open(blend_path, 'rb') as f:
            blend_artifacts = pickle.load(f)
        blend_predictor = BlendPredictor(blend_artifacts)
    
    return {
        'carrier_model': carrier_model,
        'config': config,
        'df_knn': df_knn,
        'distance_matrix': distance_matrix,
        'index_shrink_predictor': index_shrink_predictor,
        'blend_predictor': blend_predictor
    }

models = load_models()
config = models['config']
df_knn = models['df_knn']
DISTANCE_MATRIX = models['distance_matrix']
index_shrink_predictor = models['index_shrink_predictor']
blend_predictor = models['blend_predictor']

FEATURES = config['FEATURES']
ENTITY_MAPPING = config.get('ENTITY_MAPPING', 'Domestic')
DISTANCE_LOOKUP = config.get('DISTANCE_LOOKUP', {})

df_knn = df_knn[df_knn['entity_mapping'] == ENTITY_MAPPING].copy()

# ============================================
# FILTER OUT BAD/INVALID CITY NAMES FROM HISTORICAL DATA
# These are data quality issues that should not be treated as real cities
# ============================================
INVALID_CITY_NAMES = {
    'Al Farwaniyah',  # Kuwait city, not Saudi Arabia
    'Unknown',        # Placeholder/missing data
    'unknown',
    '',
    None,
    'N/A',
    'n/a',
    'NA',
    'Murjan',         # Not a real city (district/neighborhood)
}

# Filter out rows with invalid cities
df_knn = df_knn[~df_knn['pickup_city'].isin(INVALID_CITY_NAMES)]
df_knn = df_knn[~df_knn['destination_city'].isin(INVALID_CITY_NAMES)]

# Set of all cities with historical trip data
VALID_CITIES_AR = set(df_knn['pickup_city'].unique()) | set(df_knn['destination_city'].unique())

# ============================================
# STARTUP VALIDATION: Check for cities without regions
# Cities in historical data but not in normalization CSV
# ============================================
def get_cities_without_regions():
    """
    Find cities that exist in historical data but have no region defined.
    These need to be added to the city_normalization.csv file.
    """
    cities_without_region = []
    for city in VALID_CITIES_AR:
        region = get_city_region(city)
        if not region:
            cities_without_region.append(city)
    return sorted(cities_without_region)

CITIES_WITHOUT_REGIONS = get_cities_without_regions()

# ============================================
# DROPDOWN SETUP
# ============================================
all_canonicals = sorted(list(set(CITY_TO_CANONICAL.values())))
pickup_cities_en = sorted(list(set([to_english_city(c) for c in all_canonicals])))
dest_cities_en = pickup_cities_en
vehicle_types_en = sorted(set(VEHICLE_TYPE_EN.values()))
commodities = sorted(set([to_english_commodity(c) for c in df_knn['commodity'].unique()]))

# ============================================
# PRICING RULES & MODIFIERS
# ============================================
def is_core_lane(pickup_ar, dest_ar):
    """Check if lane is between main hubs (Core Lane)."""
    # Main hubs in Arabic
    CORE_CITIES = {'الرياض', 'جدة', 'الدمام'}
    
    # Normalize to ensure matching
    p_can = CITY_TO_CANONICAL.get(pickup_ar, pickup_ar)
    d_can = CITY_TO_CANONICAL.get(dest_ar, dest_ar)
    
    return (p_can in CORE_CITIES) and (d_can in CORE_CITIES)

def apply_vehicle_rules(base_price, vehicle_en, is_core, weight_tons=None):
    """
    Apply specific pricing rules based on Flatbed baseline.
    Rules:
    1. Flatbed = Baseline
    2. Curtain = +100
    3. Reefer = x1.25
    4. Lowbed = x1.85 (non-core) / x2.2 (core)
    5. Dynna = x0.7 (small) / x0.77 (large > 4T)
    6. Lorries = x0.85
    7. Closed = +50 (<=600), +100 (600-2000), +150 (>2000)
    """
    if not base_price: return None
    
    v = vehicle_en.lower()
    
    if 'curtain' in v:
        return base_price + 100
        
    elif 'refrigerated' in v or 'reefer' in v:
        return base_price * 1.25
        
    elif 'lowbed' in v:
        multiplier = 2.2 if is_core else 1.85
        return base_price * multiplier
        
    elif 'dynna' in v:
        # Threshold: 4 tons to distinguish small vs large
        multiplier = 0.77 if (weight_tons and weight_tons > 4.0) else 0.70
        return base_price * multiplier
        
    elif 'lorries' in v or 'lorry' in v:
        return base_price * 0.85
        
    elif 'closed' in v:
        if base_price <= 600: return base_price + 50
        elif base_price <= 2000: return base_price + 100
        else: return base_price + 150
        
    # Default / Flatbed
    return base_price

# ============================================
# PRICING LOGIC
# Core pricing cascade: Recency → Index+Shrinkage → Blend → Default
# ============================================
@st.cache_data
def calculate_city_cpk_stats():
    """
    Calculate inbound/outbound CPK stats per city.
    Used for backhaul probability estimation.
    
    High inbound CPK relative to outbound = good backhaul probability
    (carriers can find return loads easily)
    """
    city_stats = {}
    if len(df_knn) == 0: return {}
    
    # Outbound stats (city as pickup)
    outbound = df_knn.groupby('pickup_city').agg({'total_carrier_price': 'median', 'distance': 'median'}).reset_index()
    outbound['outbound_cpk'] = outbound['total_carrier_price'] / outbound['distance']
    
    # Inbound stats (city as destination)
    inbound = df_knn.groupby('destination_city').agg({'total_carrier_price': 'median', 'distance': 'median'}).reset_index()
    inbound['inbound_cpk'] = inbound['total_carrier_price'] / inbound['distance']
    
    for _, row in outbound.iterrows():
        if row['pickup_city'] not in city_stats: city_stats[row['pickup_city']] = {}
        city_stats[row['pickup_city']]['outbound_cpk'] = row['outbound_cpk']
    for _, row in inbound.iterrows():
        if row['destination_city'] not in city_stats: city_stats[row['destination_city']] = {}
        city_stats[row['destination_city']]['inbound_cpk'] = row['inbound_cpk']
    return city_stats

CITY_CPK_STATS = calculate_city_cpk_stats()

def get_backhaul_probability(dest_city):
    """
    Estimate backhaul probability for destination city.
    
    Based on inbound/outbound CPK ratio:
    - High ratio (>0.85): Strong inbound demand, carrier likely finds return load
    - Medium ratio (0.65-0.85): Moderate backhaul opportunity
    - Low ratio (<0.65): Weak inbound demand, carrier likely deadheads back
    
    Returns: (probability_label, margin_to_apply, ratio)
    """
    stats = CITY_CPK_STATS.get(dest_city, {})
    i, o = stats.get('inbound_cpk'), stats.get('outbound_cpk')
    if i is None or o is None or o == 0: return 'Unknown', MARGIN_UNKNOWN, None
    ratio = i / o
    if ratio >= BACKHAUL_HIGH_THRESHOLD: return 'High', MARGIN_HIGH_BACKHAUL, ratio
    elif ratio >= BACKHAUL_MEDIUM_THRESHOLD: return 'Medium', MARGIN_MEDIUM_BACKHAUL, ratio
    return 'Low', MARGIN_LOW_BACKHAUL, ratio

def calculate_rental_cost(distance_km):
    """
    Calculate rental cost index as price floor reference.
    Based on 800 SAR/day at 600 km/day.
    Rounds to nearest 0.5 day for trips > 1 day.
    """
    if not distance_km or distance_km <= 0: return None
    days = distance_km / RENTAL_KM_PER_DAY
    return round((1.0 if days < 1 else round(days * 2) / 2) * RENTAL_COST_PER_DAY, 0)

def calculate_reference_sell(pickup_ar, dest_ar, vehicle_ar, lane_data, recommended_sell):
    """
    Get reference sell price from historical shipper prices.
    Priority: Recent 90d median → Historical median → Recommended
    """
    recent = lane_data[lane_data['days_ago'] <= RECENCY_WINDOW]
    if len(recent) > 0: return round(recent['total_shipper_price'].median(), 0), 'Recent 90d'
    if len(lane_data) > 0: return round(lane_data['total_shipper_price'].median(), 0), 'Historical'
    if recommended_sell: return recommended_sell, 'Recommended'
    return None, 'N/A'

def get_distance(pickup_ar, dest_ar, lane_data=None, immediate_log=False, check_history=False):
    """
    Get distance for a route. Tries multiple sources in order:
    1. Historical trip data (median of actual trips)
    2. Reverse lane historical data
    3. Distance lookup from config
    4. Distance matrix (canonical and reverse lookups)
    
    Args:
        pickup_ar: Pickup city in Arabic
        dest_ar: Destination city in Arabic
        lane_data: Optional pre-filtered lane data
        immediate_log: If True, log errors immediately (for single-lane pricing)
                      If False, queue errors for batch write (for bulk pricing)
        check_history: If True, only log to MatchedDistances if at least one city 
                      has historical data (for Master Grid). If False, log all missing.
    
    Returns: (distance_km, source_description)
    """
    lane, rev_lane = f"{pickup_ar} → {dest_ar}", f"{dest_ar} → {pickup_ar}"
    p_can, d_can = CITY_TO_CANONICAL.get(pickup_ar, pickup_ar), CITY_TO_CANONICAL.get(dest_ar, dest_ar)
    
    # Try historical data first
    if lane_data is not None and len(lane_data[lane_data['distance'] > 0]) > 0: 
        return lane_data[lane_data['distance'] > 0]['distance'].median(), 'Historical'
    rev_data = df_knn[df_knn['lane'] == rev_lane]
    if len(rev_data[rev_data['distance'] > 0]) > 0: 
        return rev_data[rev_data['distance'] > 0]['distance'].median(), 'Historical (reverse)'
    
    # Try config lookup
    if DISTANCE_LOOKUP.get(lane): return DISTANCE_LOOKUP[lane], 'Historical'
    if DISTANCE_LOOKUP.get(rev_lane): return DISTANCE_LOOKUP[rev_lane], 'Historical (reverse)'
    
    # Try distance matrix (multiple lookup patterns)
    if (pickup_ar, dest_ar) in DISTANCE_MATRIX: return DISTANCE_MATRIX[(pickup_ar, dest_ar)], 'Matrix'
    if (p_can, d_can) in DISTANCE_MATRIX: return DISTANCE_MATRIX[(p_can, d_can)], 'Matrix (canonical)'
    if (dest_ar, pickup_ar) in DISTANCE_MATRIX: return DISTANCE_MATRIX[(dest_ar, pickup_ar)], 'Matrix (reverse)'
    if (d_can, p_can) in DISTANCE_MATRIX: return DISTANCE_MATRIX[(d_can, p_can)], 'Matrix (canonical reverse)'
    
    # Log missing distance for debugging
    pickup_en = to_english_city(pickup_ar)
    dest_en = to_english_city(dest_ar)
    
    # Log to ErrorLog sheet
    log_exception('missing_distance', {
        'pickup_city': pickup_ar, 'destination_city': dest_ar,
        'pickup_en': pickup_en, 'destination_en': dest_en
    }, immediate=immediate_log)
    
    # Also log to MatchedDistances sheet for Google Maps lookup
    if pickup_en and dest_en:
        should_log = True
        
        # If check_history is True, only log if at least one city has historical data
        if check_history:
            pickup_has_history = pickup_ar in VALID_CITIES_AR or p_can in VALID_CITIES_AR
            dest_has_history = dest_ar in VALID_CITIES_AR or d_can in VALID_CITIES_AR
            should_log = pickup_has_history or dest_has_history
        
        if should_log:
            log_missing_distance_for_lookup(pickup_ar, dest_ar, pickup_en, dest_en, immediate=immediate_log)
    
    return 0, 'Missing'

def round_to_nearest(value, nearest):
    """Round value to nearest multiple (e.g., 100 for buy price, 50 for sell price)."""
    return int(round(value / nearest) * nearest) if value and not pd.isna(value) else None

def calculate_prices(pickup_ar, dest_ar, requested_vehicle_ar, distance_km, lane_data=None, 
                     pickup_region_override=None, dest_region_override=None, weight=None):
    """
    Calculate price starting with Flatbed Baseline -> Apply Vehicle Modifiers.
    
    PRICING CASCADE (Run on Flatbed Data):
    1. RECENCY: If recent loads exist (within RECENCY_WINDOW days), use median
    2. INDEX+SHRINKAGE: If ANY historical data exists for this lane, use Index+Shrinkage model
    3. PROVINCE BLEND: For NEW lanes (no history), use 70% Province CPK + 30% City decomposition
    4. REGIONAL BLEND: If Province Blend fails (missing province data), use 5-region CPK instead
    5. CITY MULTIPLIERS: If no geographic data at all, use city multipliers only
    6. DEFAULT: Last resort - use default CPK × distance
    """
    lane = f"{pickup_ar} → {dest_ar}"
    
    # 1. FORCE FLATBED DATA for the Baseline calculation
    # We ignore the requested vehicle's historical data for the pricing model
    # to ensure we follow the "Flatbed as Baseline" rule strictly.
    flatbed_data = df_knn[(df_knn['lane'] == lane) & (df_knn['vehicle_type'] == DEFAULT_VEHICLE_AR)].copy()
    
    # Calculate BASELINE (Flatbed) Price
    recent_count = len(flatbed_data[flatbed_data['days_ago'] <= RECENCY_WINDOW])
    
    # --- PRICING CASCADE (Run on Flatbed Data) ---
    if recent_count >= 1:
        # 1. RECENCY MODEL: Use median of recent loads
        base_price = flatbed_data[flatbed_data['days_ago'] <= RECENCY_WINDOW]['total_carrier_price'].median()
        model = 'Recency (Base)'
        conf = 'High' if recent_count >= 5 else ('Medium' if recent_count >= 2 else 'Low')
        
    elif index_shrink_predictor and index_shrink_predictor.has_lane_data(lane):
        # 2. INDEX + SHRINKAGE: Lane has ANY historical data (even if not recent)
        # Uses Flatbed historical stats inside the model
        p = index_shrink_predictor.predict(pickup_ar, dest_ar, distance_km)
        base_price, model, conf = p.get('predicted_cost'), p['method'], p['confidence']
        
    elif blend_predictor:
        # 3-5. BLEND MODEL: NEW lane with NO historical data
        # Internally tries: Province CPK → Regional CPK → City Multipliers
        # Pass region overrides from coordinates if available
        p = blend_predictor.predict(pickup_ar, dest_ar, distance_km,
                                    pickup_region_override=pickup_region_override,
                                    dest_region_override=dest_region_override)
        base_price, model, conf = p.get('predicted_cost'), p['method'], p['confidence']
        
    else:
        # 6. DEFAULT: Last resort fallback (no models available)
        base_price = distance_km * 1.8 if distance_km else None
        model, conf = 'Default CPK', 'Very Low'

    # 2. APPLY VEHICLE MODIFIERS
    # Determine English name for logic check
    vehicle_en = to_english_vehicle(requested_vehicle_ar)
    is_core = is_core_lane(pickup_ar, dest_ar)
    
    final_buy_price = apply_vehicle_rules(base_price, vehicle_en, is_core, weight)
    
    # Rounding
    buy_rounded = round_to_nearest(final_buy_price, 100)
    
    # Calculate sell price (Margin logic)
    bh_prob, margin, bh_ratio = get_backhaul_probability(dest_ar)
    sell_rounded = round_to_nearest(buy_rounded * (1 + margin), 50) if buy_rounded else None
    
    # Get reference sell (Visual only - using original requested vehicle data if available)
    # We pass the ORIGINAL lane_data (if provided) to show history for the actual requested truck
    req_lane_data = lane_data if lane_data is not None else df_knn[(df_knn['lane'] == lane) & (df_knn['vehicle_type'] == requested_vehicle_ar)].copy()
    ref_sell, ref_src = calculate_reference_sell(pickup_ar, dest_ar, requested_vehicle_ar, req_lane_data, sell_rounded)
    
    return {
        'buy_price': buy_rounded, 
        'sell_price': sell_rounded, 
        'base_flatbed_price': base_price,  # Helpful for debug
        'ref_sell_price': round_to_nearest(ref_sell, 50),
        'ref_sell_source': ref_src, 
        'rental_cost': calculate_rental_cost(distance_km),
        'target_margin': f"{margin:.0%}", 
        'backhaul_probability': bh_prob, 
        'backhaul_ratio': round(bh_ratio, 2) if bh_ratio else None,
        'model_used': f"{model} + {vehicle_en} Rule", 
        'confidence': conf, 
        'recent_count': recent_count
    }

def select_spaced_samples(df, n_samples):
    if len(df) == 0: return pd.DataFrame()
    df = df.sort_values('days_ago')
    selected = []
    last_day = -MIN_DAYS_APART
    
    for _, row in df[df['days_ago'] <= RECENT_PRIORITY_DAYS].iterrows():
        if row['days_ago'] >= last_day + MIN_DAYS_APART:
            selected.append(row)
            last_day = row['days_ago']
            if len(selected) >= n_samples: break
            
    if len(selected) < n_samples:
        for _, row in df[df['days_ago'] > RECENT_PRIORITY_DAYS].iterrows():
            if row['days_ago'] >= last_day + MIN_DAYS_APART:
                selected.append(row)
                last_day = row['days_ago']
                if len(selected) >= n_samples: break
    return pd.DataFrame(selected)

def get_ammunition_loads(lane, vehicle_ar, commodity=None):
    matches = df_knn[(df_knn['lane'] == lane) & (df_knn['vehicle_type'] == vehicle_ar) & (df_knn['days_ago'] <= MAX_AGE_DAYS)].copy()
    if len(matches) == 0: return pd.DataFrame(), pd.DataFrame()
    
    if commodity and commodity not in ['', 'Auto', 'auto', None]:
        same, other = matches[matches['commodity'] == commodity], matches[matches['commodity'] != commodity]
    else:
        mode = matches['commodity'].mode().iloc[0] if len(matches) > 0 else None
        same, other = (matches[matches['commodity'] == mode], matches[matches['commodity'] != mode]) if mode else (pd.DataFrame(), pd.DataFrame())
    return select_spaced_samples(same, 5), select_spaced_samples(other, 5)

def price_single_route(pickup_ar, dest_ar, vehicle_ar=None, commodity=None, weight=None):
    if not vehicle_ar or vehicle_ar in ['', 'Auto', 'auto']: vehicle_ar = DEFAULT_VEHICLE_AR
    lane = f"{pickup_ar} → {dest_ar}"
    lane_data = df_knn[(df_knn['lane'] == lane) & (df_knn['vehicle_type'] == vehicle_ar)].copy()
    
    if commodity and commodity not in ['', 'Auto', 'auto', None]:
        if len(lane_data[lane_data['commodity'] == commodity]) > 0: lane_data = lane_data[lane_data['commodity'] == commodity]
    
    rec_data = lane_data[lane_data['days_ago'] <= RECENCY_WINDOW]
    
    def get_stats(d, col):
        if len(d) == 0: return None, None, None, 0
        return round(d[col].min(),0), round(d[col].median(),0), round(d[col].max(),0), len(d)
    
    h_min, h_med, h_max, h_count = get_stats(lane_data, 'total_carrier_price')
    r_min, r_med, r_max, r_count = get_stats(rec_data, 'total_carrier_price')
    hs_min, hs_med, hs_max, _ = get_stats(lane_data, 'total_shipper_price')
    rs_min, rs_med, rs_max, _ = get_stats(rec_data, 'total_shipper_price')
    
    if not commodity or commodity in ['', 'Auto']:
        commodity = lane_data['commodity'].mode().iloc[0] if len(lane_data) > 0 else df_knn['commodity'].mode().iloc[0]
    
    if not weight:
        w = df_knn[(df_knn['commodity'] == commodity) & (df_knn['weight'] > 0)]['weight']
        weight = w.median() if len(w) > 0 else 25.0
        
    # Use immediate_log=True for single-lane pricing (writes to sheets immediately)
    dist, dist_src = get_distance(pickup_ar, dest_ar, lane_data, immediate_log=True)
    if dist == 0: dist, dist_src = 500, 'Default'
    
    pricing = calculate_prices(pickup_ar, dest_ar, vehicle_ar, dist, lane_data, weight=weight)
    
    res = {
        'Vehicle_Type': to_english_vehicle(vehicle_ar), 'Commodity': to_english_commodity(commodity),
        'Weight_Tons': round(weight, 1), 'Distance_km': round(dist, 0), 'Distance_Source': dist_src,
        'Hist_Count': h_count, 'Hist_Min': h_min, 'Hist_Median': h_med, 'Hist_Max': h_max,
        f'Recent_{RECENCY_WINDOW}d_Count': r_count, f'Recent_{RECENCY_WINDOW}d_Min': r_min, f'Recent_{RECENCY_WINDOW}d_Median': r_med, f'Recent_{RECENCY_WINDOW}d_Max': r_max,
        'Hist_Sell_Min': hs_min, 'Hist_Sell_Median': hs_med, 'Hist_Sell_Max': hs_max,
        f'Recent_{RECENCY_WINDOW}d_Sell_Min': rs_min, f'Recent_{RECENCY_WINDOW}d_Sell_Median': rs_med, f'Recent_{RECENCY_WINDOW}d_Sell_Max': rs_max,
        'Buy_Price': pricing['buy_price'], 'Rec_Sell_Price': pricing['sell_price'],
        'Ref_Sell_Price': pricing['ref_sell_price'], 'Ref_Sell_Source': pricing['ref_sell_source'],
        'Rental_Cost': pricing['rental_cost'], 'Target_Margin': pricing['target_margin'],
        'Backhaul_Probability': pricing['backhaul_probability'], 'Backhaul_Ratio': pricing['backhaul_ratio'],
        'Model_Used': pricing['model_used'], 'Confidence': pricing['confidence'],
        'Cost_Per_KM': round(pricing['buy_price']/dist, 2) if pricing['buy_price'] and dist > 0 else None,
        'Is_Rare_Lane': r_count == 0
    }
    
    if index_shrink_predictor:
        p = index_shrink_predictor.predict(pickup_ar, dest_ar, dist)
        res.update({'IndexShrink_Price': p.get('predicted_cost'), 'IndexShrink_Upper': p.get('cost_high'), 'IndexShrink_Method': p['method']})
    if blend_predictor:
        p = blend_predictor.predict(pickup_ar, dest_ar, dist)
        res.update({'Blend_Price': p.get('predicted_cost'), 'Blend_Upper': p.get('cost_high'), 'Blend_Method': p['method']})
    return res

def lookup_route_stats(pickup_ar, dest_ar, vehicle_ar=None, dist_override=None, check_history=False,
                       pickup_region_override=None, dest_region_override=None, weight=None):
    """
    Lookup route stats for bulk pricing - uses batch logging (immediate_log=False).
    
    Args:
        pickup_ar: Pickup city in Arabic
        dest_ar: Destination city in Arabic
        vehicle_ar: Vehicle type in Arabic (optional)
        dist_override: If provided, use this distance instead of looking it up
        check_history: If True, only log to MatchedDistances if at least one city 
                      has historical data (for Master Grid)
        pickup_region_override: If provided, use this region for pickup (from coordinates)
        dest_region_override: If provided, use this region for destination (from coordinates)
        weight: Weight in tons (optional, used for vehicle sizing logic)
    """
    if not vehicle_ar or vehicle_ar in ['', 'Auto', 'auto']: vehicle_ar = DEFAULT_VEHICLE_AR
    lane_data = df_knn[(df_knn['lane'] == f"{pickup_ar} → {dest_ar}") & (df_knn['vehicle_type'] == vehicle_ar)].copy()
    
    # Use override distance if provided, otherwise look it up
    if dist_override is not None and dist_override > 0:
        dist = dist_override
        dist_source = 'User Provided'
    else:
        # Use immediate_log=False for bulk pricing (queues errors for batch write)
        dist, dist_source = get_distance(pickup_ar, dest_ar, lane_data, immediate_log=False, check_history=check_history)
    
    pricing = calculate_prices(pickup_ar, dest_ar, vehicle_ar, dist, lane_data,
                               pickup_region_override=pickup_region_override,
                               dest_region_override=dest_region_override,
                               weight=weight)
    return {
        'From': to_english_city(pickup_ar), 
        'To': to_english_city(dest_ar), 
        'Vehicle': to_english_vehicle(vehicle_ar), 
        'Weight_Tons': weight if weight else 25.0,
        'Distance': int(dist) if dist else 0,
        'Distance_Source': dist_source,
        'Buy_Price': pricing['buy_price'], 'Rec_Sell': pricing['sell_price'],
        'Ref_Sell': pricing['ref_sell_price'], 'Ref_Sell_Src': pricing['ref_sell_source'],
        'Rental_Cost': pricing['rental_cost'], 'Margin': pricing['target_margin'],
        'Model': pricing['model_used'], 'Confidence': pricing['confidence'], 'Recent_N': pricing['recent_count']
    }
                           
# ============================================
# APP UI
# ============================================
st.title("🚚 Freight Pricing Tool")
model_status = []
if index_shrink_predictor: model_status.append("✅ Index+Shrinkage")
if blend_predictor:
    # Check if using provinces (13) or regions (5)
    if blend_predictor.using_provinces:
        model_status.append("✅ Blend 0.7 (Province)")
    else:
        model_status.append("✅ Blend 0.7 (Region)")
dist_status = f"✅ {len(DISTANCE_MATRIX):,} distances" if DISTANCE_MATRIX else ""
st.caption(f"ML-powered pricing | Domestic | {' | '.join(model_status)} | {dist_status}")

# Warning for cities in historical data but not in normalization CSV
if CITIES_WITHOUT_REGIONS:
    with st.expander(f"⚠️ {len(CITIES_WITHOUT_REGIONS)} cities in historical data missing from canonicals CSV", expanded=False):
        st.warning("""
        **Action Required:** These cities exist in the historical trip data but are NOT in the city_normalization.csv file.
        They need to be added with their canonical form and region.
        """)
        for city in CITIES_WITHOUT_REGIONS:
            st.code(city)
        st.info("Add these to city_normalization.csv with columns: variant, canonical, english, region")

tab1, tab2, tab3 = st.tabs(["🎯 Single Route Pricing", "📦 Bulk Route Lookup","🗺️ Map Explorer"])

with tab1:
    st.subheader("📋 Route Information")
    
    # Callback for swap - must be defined before the button
    def swap_cities():
        pickup = st.session_state.single_pickup
        dest = st.session_state.single_dest
        st.session_state.single_pickup = dest
        st.session_state.single_dest = pickup
    
    if not pickup_cities_en:
         st.error("City list is empty. Check normalization file.")
         st.stop()
    
    # Initialize defaults in session state if not present (avoids warning on swap)
    if 'single_pickup' not in st.session_state:
        st.session_state.single_pickup = 'Jeddah' if 'Jeddah' in pickup_cities_en else pickup_cities_en[0]
    if 'single_dest' not in st.session_state:
        st.session_state.single_dest = 'Riyadh' if 'Riyadh' in dest_cities_en else dest_cities_en[0]
    if 'single_vehicle' not in st.session_state:
        st.session_state.single_vehicle = DEFAULT_VEHICLE_EN if DEFAULT_VEHICLE_EN in vehicle_types_en else vehicle_types_en[0]
    
    # Use columns with swap button in the middle
    col1, col_swap, col2, col3 = st.columns([3, 0.5, 3, 3])
    with col1:
        pickup_en = st.selectbox("Pickup City", options=pickup_cities_en, key='single_pickup')
        pickup_city = to_arabic_city(pickup_en)
    with col_swap:
        st.markdown("<br>", unsafe_allow_html=True)  # Align with selectbox
        st.button("⇄", key='swap_cities', on_click=swap_cities, help="Swap Pickup ↔ Destination")
    with col2:
        dest_en = st.selectbox("Destination City", options=dest_cities_en, key='single_dest')
        destination_city = to_arabic_city(dest_en)
    with col3:
        vehicle_en = st.selectbox("Vehicle Type", options=vehicle_types_en, key='single_vehicle')
        vehicle_type = to_arabic_vehicle(vehicle_en)

    st.subheader("📦 Optional Details")
    col1, col2, col3 = st.columns(3)
    with col1: st.selectbox("Container", options=['Auto-detect', 'Yes (Container)', 'No (Non-container)'], key='single_container')
    with col2:
        comm_opt = ['Auto-detect'] + commodities
        comm_sel = st.selectbox("Commodity", options=comm_opt, key='single_commodity')
        comm_in = None if comm_sel == 'Auto-detect' else to_arabic_commodity(comm_sel)
    with col3:
        w = st.number_input("Weight (Tons)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, help="0 = Auto-detect", key='single_weight')
        weight = None if w == 0 else w

    st.markdown("---")
    if st.button("🎯 Generate Pricing", type="primary", use_container_width=True, key='single_generate'):
        result = price_single_route(pickup_city, destination_city, vehicle_type, comm_in, weight)
        st.session_state.last_result, st.session_state.last_lane = result, {'pickup_ar': pickup_city, 'dest_ar': destination_city, 'pickup_en': pickup_en, 'dest_en': dest_en}
        
        if result['Distance_km'] == 0 or result['Distance_Source'] == 'Default': st.error(f"⚠️ Distance data missing or estimated ({result['Distance_km']} km)")
        
        st.header("💰 Pricing Results")
        st.info(f"**{pickup_en} → {dest_en}** | 🚛 {result['Vehicle_Type']} | 📏 {result['Distance_km']:.0f} km | ⚖️ {result['Weight_Tons']:.1f} T")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🛒 BUY PRICE", f"{result['Buy_Price']:,} SAR" if result['Buy_Price'] else "N/A")
        c2.metric("💵 SELL PRICE", f"{result['Rec_Sell_Price']:,} SAR" if result.get('Rec_Sell_Price') else "N/A", help="Buy + Margin")
        c3.metric("📊 Target Margin", result['Target_Margin'])
        c4.metric("📊 Confidence", f"{ {'High':'🟢','Medium':'🟡','Low':'🟠','Very Low':'🔴'}.get(result['Confidence'],'⚪') } {result['Confidence']}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🚛 Rental Index", f"{result['Rental_Cost']:,.0f} SAR" if result['Rental_Cost'] else "N/A")
        c2.metric("📈 REF. SELL", f"{result['Ref_Sell_Price']:,} SAR" if result['Ref_Sell_Price'] else "N/A", help=f"Source: {result.get('Ref_Sell_Source')}")
        c3.empty()
        c4.metric("🔄 Backhaul", f"{ {'High':'🟢','Medium':'🟡','Low':'🔴','Unknown':'⚪'}.get(result['Backhaul_Probability'],'⚪') } {result['Backhaul_Probability']}")
        
        st.caption(f"Model: **{result['Model_Used']}** | CPK: {result['Cost_Per_KM']} SAR/km | Ref. Sell Source: {result.get('Ref_Sell_Source', 'N/A')}")
        if result['Is_Rare_Lane']: st.warning(f"⚠️ **Rare Lane** - No recent loads. Using model prediction.")

        st.markdown("---")
        st.subheader("📊 Price History")
        hc, rc = result['Hist_Count'], result[f'Recent_{RECENCY_WINDOW}d_Count']
        
        if hc > 0 or rc > 0:
            c1, c2 = st.columns(2)
            # Recent data on the LEFT (more actionable)
            with c1:
                st.markdown(f"### Recent ({RECENCY_WINDOW}d)")
                st.dataframe(pd.DataFrame({
                    'Metric': ['Count', 'Min', 'Median', 'Max'],
                    'Buy Price': [f"{rc} loads" if rc else "—", f"{result[f'Recent_{RECENCY_WINDOW}d_Min']:,} SAR" if result[f'Recent_{RECENCY_WINDOW}d_Min'] else "—", f"{result[f'Recent_{RECENCY_WINDOW}d_Median']:,} SAR" if result[f'Recent_{RECENCY_WINDOW}d_Median'] else "—", f"{result[f'Recent_{RECENCY_WINDOW}d_Max']:,} SAR" if result[f'Recent_{RECENCY_WINDOW}d_Max'] else "—"],
                    'Sell Price': [f"{rc} loads" if rc else "—", f"{result[f'Recent_{RECENCY_WINDOW}d_Sell_Min']:,} SAR" if result[f'Recent_{RECENCY_WINDOW}d_Sell_Min'] else "—", f"{result[f'Recent_{RECENCY_WINDOW}d_Sell_Median']:,} SAR" if result[f'Recent_{RECENCY_WINDOW}d_Sell_Median'] else "—", f"{result[f'Recent_{RECENCY_WINDOW}d_Sell_Max']:,} SAR" if result[f'Recent_{RECENCY_WINDOW}d_Sell_Max'] else "—"]
                }), use_container_width=True, hide_index=True)
            # Historical data on the RIGHT (context/reference)
            with c2:
                st.markdown("### Historical")
                st.dataframe(pd.DataFrame({
                    'Metric': ['Count', 'Min', 'Median', 'Max'],
                    'Buy Price': [f"{hc} loads" if hc else "—", f"{result['Hist_Min']:,} SAR" if result['Hist_Min'] else "—", f"{result['Hist_Median']:,} SAR" if result['Hist_Median'] else "—", f"{result['Hist_Max']:,} SAR" if result['Hist_Max'] else "—"],
                    'Sell Price': [f"{hc} loads" if hc else "—", f"{result['Hist_Sell_Min']:,} SAR" if result['Hist_Sell_Min'] else "—", f"{result['Hist_Sell_Median']:,} SAR" if result['Hist_Sell_Median'] else "—", f"{result['Hist_Sell_Max']:,} SAR" if result['Hist_Sell_Max'] else "—"]
                }), use_container_width=True, hide_index=True)
        else: st.warning("No historical or recent data available")
        
        st.markdown("---")
        with st.expander("🔮 Model Predictions Comparison", expanded=False):
            model_df = []
            if 'IndexShrink_Price' in result and result['IndexShrink_Price']:
                model_df.append({'Model': result['IndexShrink_Method'], 'Prediction': f"{result['IndexShrink_Price']:,.0f} SAR", 'Upper Bound': f"{result['IndexShrink_Upper']:,.0f} SAR"})
            if 'Blend_Price' in result and result['Blend_Price']:
                model_df.append({'Model': result['Blend_Method'], 'Prediction': f"{result['Blend_Price']:,.0f} SAR", 'Upper Bound': f"{result['Blend_Upper']:,.0f} SAR"})
            if model_df: st.dataframe(pd.DataFrame(model_df), use_container_width=True, hide_index=True)
            
        if not result['Is_Rare_Lane']:
            st.markdown("---")
            st.subheader("🚚 Your Ammunition (Recent Matches)")
            
            lane_ar = f"{pickup_city} → {destination_city}"
            same_samples, other_samples = get_ammunition_loads(lane_ar, vehicle_type, comm_in)
            
            # Check if shipper/company column exists in data
            has_shipper = 'shipper_name' in df_knn.columns or 'company' in df_knn.columns or 'client' in df_knn.columns
            shipper_col = 'shipper_name' if 'shipper_name' in df_knn.columns else ('company' if 'company' in df_knn.columns else 'client')
            
            commodity_used = to_english_commodity(comm_in) if comm_in else result['Commodity']
            if len(same_samples) > 0:
                st.markdown(f"**Same Commodity ({commodity_used}):**")
                same_samples['Lane_EN'] = same_samples['pickup_city'].apply(to_english_city) + ' → ' + same_samples['destination_city'].apply(to_english_city)
                same_samples['Commodity_EN'] = same_samples['commodity'].apply(to_english_commodity)
                
                # Build columns list based on available data
                if has_shipper and shipper_col in same_samples.columns:
                    display_same = same_samples[['pickup_date', 'Lane_EN', 'Commodity_EN', shipper_col, 'total_carrier_price', 'days_ago']].copy()
                    display_same.columns = ['Date', 'Lane', 'Commodity', 'Shipper', 'Carrier (SAR)', 'Days Ago']
                else:
                    display_same = same_samples[['pickup_date', 'Lane_EN', 'Commodity_EN', 'total_carrier_price', 'days_ago']].copy()
                    display_same.columns = ['Date', 'Lane', 'Commodity', 'Carrier (SAR)', 'Days Ago']
                
                display_same['Date'] = pd.to_datetime(display_same['Date']).dt.strftime('%Y-%m-%d')
                display_same['Carrier (SAR)'] = display_same['Carrier (SAR)'].round(0).astype(int)
                st.dataframe(display_same, use_container_width=True, hide_index=True)
            else:
                st.caption(f"No recent loads with {commodity_used}")
            
            if len(other_samples) > 0:
                st.markdown("**Other Commodities:**")
                other_samples['Lane_EN'] = other_samples['pickup_city'].apply(to_english_city) + ' → ' + other_samples['destination_city'].apply(to_english_city)
                other_samples['Commodity_EN'] = other_samples['commodity'].apply(to_english_commodity)
                
                # Build columns list based on available data
                if has_shipper and shipper_col in other_samples.columns:
                    display_other = other_samples[['pickup_date', 'Lane_EN', 'Commodity_EN', shipper_col, 'total_carrier_price', 'days_ago']].copy()
                    display_other.columns = ['Date', 'Lane', 'Commodity', 'Shipper', 'Carrier (SAR)', 'Days Ago']
                else:
                    display_other = other_samples[['pickup_date', 'Lane_EN', 'Commodity_EN', 'total_carrier_price', 'days_ago']].copy()
                    display_other.columns = ['Date', 'Lane', 'Commodity', 'Carrier (SAR)', 'Days Ago']
                
                display_other['Date'] = pd.to_datetime(display_other['Date']).dt.strftime('%Y-%m-%d')
                display_other['Carrier (SAR)'] = display_other['Carrier (SAR)'].round(0).astype(int)
                st.dataframe(display_other, use_container_width=True, hide_index=True)
            else:
                st.caption("No recent loads with other commodities")
            
            total_shown = len(same_samples) + len(other_samples)
            if total_shown > 0:
                st.caption(f"**{total_shown} loads** | Samples ≥{MIN_DAYS_APART} days apart | Max {MAX_AGE_DAYS} days old")

    if 'last_result' in st.session_state and 'last_lane' in st.session_state:
        st.markdown("---")
        
        # Distance Suggestion Section
        with st.expander("📏 Suggest Distance Change", expanded=False):
            l, r = st.session_state.last_lane, st.session_state.last_result
            st.caption(f"Lane: **{l['pickup_en']} → {l['dest_en']}** | Current distance: **{r['Distance_km']:.0f} km**")
            
            col1, col2 = st.columns(2)
            with col1:
                user_name = st.text_input("Your Name", key='suggest_user_name', placeholder="Enter your name")
            with col2:
                suggested_dist = st.number_input(
                    "Suggested Distance", 
                    min_value=0.0, 
                    max_value=5000.0, 
                    value=float(r['Distance_km']) if r['Distance_km'] > 0 else 0.0,
                    step=10.0,
                    key='suggest_distance',
                    help="Enter distance in km (no need to write 'km')"
                )
            
            if st.button("✅ Submit Distance Suggestion", type="primary", key='submit_distance_suggestion'):
                if not user_name or user_name.strip() == '':
                    st.error("Please enter your name")
                elif suggested_dist <= 0:
                    st.error("Please enter a valid distance greater than 0")
                elif suggested_dist == r['Distance_km']:
                    st.warning("Suggested distance is the same as current distance")
                else:
                    success, message = suggest_distance_change(
                        l['pickup_ar'], l['dest_ar'], 
                        l['pickup_en'], l['dest_en'],
                        r['Distance_km'], suggested_dist, 
                        user_name.strip()
                    )
                    if success:
                        st.success(f"✅ {message}")
                        st.info(f"Changed: {r['Distance_km']:.0f} km → {suggested_dist:.0f} km")
                    else:
                        st.error(f"❌ {message}")
        
        # Report Issue Section (existing)
        with st.expander("🚨 Report Other Issue", expanded=False):
            st.caption(f"Reporting: {st.session_state.last_lane['pickup_en']} → {st.session_state.last_lane['dest_en']}")
            iss = st.selectbox("Issue Type", ["Data wrong", "Price seems off", "Other"], key='rep_type')
            note = st.text_area("Notes", key='rep_note')
            if st.button("Submit Report"):
                l, r = st.session_state.last_lane, st.session_state.last_result
                if report_distance_issue(l['pickup_ar'], l['dest_ar'], l['pickup_en'], l['dest_en'], r['Distance_km'], iss, note): st.success("Report submitted")
                else: st.warning("Report failed (GS not configured)")

with tab2:
    st.subheader("📦 Bulk Route Lookup")
    
    # ============================================
    # MULTI-STEP WIZARD FOR BULK LOOKUP
    # Step 0: Upload & Username
    # Step 1: City Resolution (Fuzzy Matching)
    # Step 2: Distance Review
    # Step 3: Final Pricing
    # ============================================
    
    # Initialize wizard state
    if 'bulk_wizard_step' not in st.session_state:
        st.session_state.bulk_wizard_step = 0
    if 'bulk_wizard_data' not in st.session_state:
        st.session_state.bulk_wizard_data = {}
    
    def reset_wizard():
        """Reset the wizard to initial state"""
        st.session_state.bulk_wizard_step = 0
        st.session_state.bulk_wizard_data = {}
        # Clear related session state
        for key in list(st.session_state.keys()):
            if key.startswith('bulk_') and key not in ['bulk_wizard_step', 'bulk_wizard_data']:
                del st.session_state[key]
    
    # Show current step indicator
    steps = ["📤 Upload", "🏙️ Cities", "📏 Distances", "💰 Pricing"]
    current_step = st.session_state.bulk_wizard_step
    
    # Progress indicator
    cols = st.columns(len(steps))
    for i, (col, step_name) in enumerate(zip(cols, steps)):
        with col:
            if i < current_step:
                st.success(f"✅ {step_name}")
            elif i == current_step:
                st.info(f"➡️ {step_name}")
            else:
                st.caption(f"⬜ {step_name}")
    
    st.markdown("---")
    
    # ============================================
    # STEP 0: Upload & Username
    # ============================================
    if current_step == 0:
        st.markdown("### Step 1: Upload CSV & Enter Your Name")
        
        st.markdown("""
        **Columns (by position, names ignored):**
        1. **Pickup City** (required)
        2. **Destination City** (required)
        3. **Distance** (optional - overrides lookup)
        4. **Vehicle Type** (optional - default: Flatbed Trailer)
        """)
        
        # Username input
        username = st.text_input("👤 Your Name", key='bulk_username', 
                                  placeholder="Enter your name for tracking",
                                  help="Your name will be logged with any changes you make")
        
        # Template button
        if st.button("📋 Open Template in Google Sheets"):
            template_df = pd.DataFrame({
                'Pickup': ['Jeddah', 'Riyadh', 'Unknown City'], 
                'Destination': ['Riyadh', 'Dammam', 'Jeddah'], 
                'Distance': ['', '', '450'],
                'Vehicle_Type': ['Flatbed Trailer', '', '']
            })
            success, msg = populate_rfq_template_sheet(template_df)
            if success:
                rfq_url = st.secrets.get('RFQ_url', '')
                st.success(msg)
                if rfq_url:
                    st.markdown(f"[🔗 Open Template Sheet]({rfq_url})")
            else:
                st.error(msg)
        
        # File uploader
        upl = st.file_uploader("Upload CSV", type=['csv'], key='bulk_csv_upload')
        
        if upl:
            try:
                r_df = pd.read_csv(upl)
                st.success(f"✅ Loaded {len(r_df)} routes")
                
                with st.expander("📋 Preview Data"):
                    st.dataframe(r_df.head(10), use_container_width=True)
                
                # Vehicle type selection
                st.markdown("##### 🚛 Vehicle Types")
                st.caption("Select vehicle types to price. Multiple selections will create multiple pricing rows per route.")
                
                v_cols = st.columns(4)
                selected_vehicles = []
                for i, v_en in enumerate(vehicle_types_en):
                    with v_cols[i % 4]:
                        if st.checkbox(v_en, value=(v_en == 'Flatbed Trailer'), key=f'vtype_{v_en}'):
                            selected_vehicles.append(v_en)
                
                if not selected_vehicles:
                    st.warning("Please select at least one vehicle type")
                
                # Proceed button
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button("▶️ Analyze Cities", type="primary", use_container_width=True,
                                disabled=not username or not selected_vehicles):
                        if not username:
                            st.error("Please enter your name")
                        elif not selected_vehicles:
                            st.error("Please select at least one vehicle type")
                        else:
                            # Parse the CSV and find unmatched cities
                            unmatched_cities = {}  # {original_name: {rows: [...], col: 'Pickup'/'Destination'}}
                            parsed_rows = []
                            
                            for i in range(len(r_df)):
                                row = r_df.iloc[i]
                                p_raw = str(row.iloc[0]).strip() if len(row) > 0 else ''
                                d_raw = str(row.iloc[1]).strip() if len(row) > 1 else ''
                                
                                # Parse distance
                                dist_override = None
                                if len(row) > 2:
                                    try:
                                        val = row.iloc[2]
                                        if pd.notna(val) and str(val).strip() not in ['', 'nan', 'None']:
                                            dist_override = float(str(val).replace(',', '').strip())
                                    except (ValueError, TypeError):
                                        pass
                                
                                # Parse vehicle
                                v_raw = ''
                                if len(row) > 3:
                                    v_raw = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ''
                                
                                # Normalize cities
                                p_ar, p_ok = normalize_city(p_raw)
                                d_ar, d_ok = normalize_city(d_raw)
                                
                                # Track unmatched
                                if not p_ok and p_raw:
                                    if p_raw not in unmatched_cities:
                                        unmatched_cities[p_raw] = {'rows': [], 'col': 'Pickup'}
                                    unmatched_cities[p_raw]['rows'].append(i + 1)
                                    
                                    # =======================================================
                                    # 👇 FIX 1 START: Explicitly log Pickup Error
                                    # =======================================================
                                    log_exception('unmatched_city', {
                                        'pickup_city': p_raw,      # Force value to Column C
                                        'destination_city': '',    # Force Column D empty
                                        'original_value': p_raw, 
                                        'column': 'Pickup', 
                                        'row_num': i + 1
                                    }, immediate=False)
                                    # =======================================================
                                    # 👆 FIX 1 END
                                    # =======================================================
                                
                                if not d_ok and d_raw:
                                    if d_raw not in unmatched_cities:
                                        unmatched_cities[d_raw] = {'rows': [], 'col': 'Destination'}
                                    unmatched_cities[d_raw]['rows'].append(i + 1)
                                
                                    # =======================================================
                                    # 👇 FIX 1 START: Explicitly log Destination Error
                                    # =======================================================
                                    log_exception('unmatched_city', {
                                        'pickup_city': '',         # Force Column C empty
                                        'destination_city': d_raw, # Force value to Column D
                                        'original_value': d_raw, 
                                        'column': 'Destination', 
                                        'row_num': i + 1
                                    }, immediate=False)
                                    # =======================================================
                                    # 👆 FIX 1 END
                                    # =======================================================
                                
                                parsed_rows.append({
                                    'row_num': i + 1,
                                    'pickup_raw': p_raw,
                                    'dest_raw': d_raw,
                                    'pickup_ar': p_ar,
                                    'dest_ar': d_ar,
                                    'pickup_ok': p_ok,
                                    'dest_ok': d_ok,
                                    'dist_override': dist_override,
                                    'vehicle_raw': v_raw
                                })
                            
                            # Log missing distances EARLY for matched city pairs
                            # This allows Google Sheets to start calculating while user resolves cities
                            matched_pairs_checked = set()
                            for row in parsed_rows:
                                if row['pickup_ok'] and row['dest_ok'] and not row.get('dist_override'):
                                    pair_key = (row['pickup_ar'], row['dest_ar'])
                                    if pair_key not in matched_pairs_checked:
                                        matched_pairs_checked.add(pair_key)
                                        # Check if distance exists, log if missing
                                        dist, source = get_distance(
                                            row['pickup_ar'], row['dest_ar'], 
                                            immediate_log=False
                                        )
                            
                            # =======================================================
                            # 👇 FIX 2 START: Flush logs to sheets BEFORE wizard moves on
                            # =======================================================
                            dist_flushed_ok, dist_flushed_count = flush_matched_distances_to_sheet()
                            err_flushed_ok, err_flushed_count = flush_error_log_to_sheet()
                            
                            # Feedback Logic to show user what happened
                            msgs = []
                            if dist_flushed_count > 0:
                                msgs.append(f"✅ Logged {dist_flushed_count} missing distances")
                            if err_flushed_count > 0:
                                msgs.append(f"⚠️ Logged {err_flushed_count} errors to sheet")
                                
                            if msgs:
                                st.session_state['last_distance_flush'] = " | ".join(msgs)
                            elif not dist_flushed_ok or not err_flushed_ok:
                                st.session_state['last_distance_flush'] = "⚠️ Failed to flush logs to sheets"
                            else:
                                st.session_state['last_distance_flush'] = "ℹ️ No new issues to log"
                            # =======================================================
                            # 👆 FIX 2 END
                            # =======================================================
                            
                            # Store in session state
                            st.session_state.bulk_wizard_data = {
                                'username': username,
                                'selected_vehicles': selected_vehicles,
                                'parsed_rows': parsed_rows,
                                'unmatched_cities': unmatched_cities,
                                'original_df': r_df
                            }
                            
                            # Move to next step
                            if unmatched_cities:
                                # Run fuzzy matching for all unmatched
                                if RAPIDFUZZ_AVAILABLE:
                                    fuzzy_results = batch_fuzzy_match_cities(list(unmatched_cities.keys()), threshold=80)
                                    st.session_state.bulk_wizard_data['fuzzy_results'] = fuzzy_results
                                st.session_state.bulk_wizard_step = 1
                            else:
                                # No unmatched cities, skip to distance review
                                st.session_state.bulk_wizard_step = 2
                            st.rerun()
                
                with col2:
                    if st.button("🔄 Reset", use_container_width=True):
                        reset_wizard()
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    
    # ============================================
    # STEP 1: City Resolution
    # ============================================
    elif current_step == 1:
        st.markdown("### Step 2: Resolve Unmatched Cities")
        
        # Show last distance flush result if any
        if 'last_distance_flush' in st.session_state:
            st.caption(st.session_state['last_distance_flush'])
        
        # Show any flush errors
        if 'flush_errors' in st.session_state and st.session_state.flush_errors:
            for err in st.session_state.flush_errors:
                st.warning(err)
            st.session_state.flush_errors = []  # Clear after showing
        
        wizard_data = st.session_state.bulk_wizard_data
        unmatched = wizard_data.get('unmatched_cities', {})
        fuzzy_results = wizard_data.get('fuzzy_results', {})
        username = wizard_data.get('username', '')
        
        if not unmatched:
            st.success("All cities matched! Proceeding to distance review...")
            st.session_state.bulk_wizard_step = 2
            st.rerun()
        
        st.info(f"Found **{len(unmatched)}** unique unmatched city names across your routes.")
        
        # Initialize resolution state if not exists
        if 'city_resolutions' not in st.session_state:
            st.session_state.city_resolutions = {}
        
        # Build resolution table
        st.markdown("#### Review and resolve each city:")
        st.caption("Accept the suggested match, provide coordinates for a new city, or ignore to skip these routes.")
        
        for idx, (city_name, info) in enumerate(unmatched.items()):
            with st.expander(f"🏙️ **{city_name}** (appears in {len(info['rows'])} rows)", expanded=True):
                fuzzy = fuzzy_results.get(city_name, {})
                
                # Resolution action selector
                current_resolution = st.session_state.city_resolutions.get(city_name, {})
                current_type = current_resolution.get('type', 'pending')
                
                action_options = ["🔍 Resolve", "⏭️ Ignore (skip routes)"]
                action_idx = 1 if current_type == 'ignored' else 0
                
                action = st.radio(
                    "Action",
                    options=action_options,
                    index=action_idx,
                    key=f"action_{idx}",
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                if action == "⏭️ Ignore (skip routes)":
                    st.warning(f"⚠️ {len(info['rows'])} routes with this city will be **excluded** from pricing.")
                    st.session_state.city_resolutions[city_name] = {
                        'type': 'ignored',
                        'rows': info['rows']
                    }
                else:
                    # Clear ignored status if switching back
                    if current_type == 'ignored':
                        del st.session_state.city_resolutions[city_name]
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.caption(f"Column: {info['col']} | Rows: {', '.join(map(str, info['rows'][:5]))}{'...' if len(info['rows']) > 5 else ''}")
                        
                        # Show fuzzy match suggestion
                        if fuzzy.get('match_found'):
                            confidence = fuzzy.get('confidence', 0)
                            suggested = fuzzy.get('suggested_canonical', '')
                            suggested_en = to_english_city(suggested)
                            
                            if confidence >= 90:
                                st.success(f"✅ High confidence match: **{suggested_en}** ({confidence}%)")
                            else:
                                st.warning(f"⚠️ Possible match: **{suggested_en}** ({confidence}%)")
                            
                            # Show other matches if available
                            all_matches = fuzzy.get('all_matches', [])
                            if len(all_matches) > 1:
                                with st.expander("Other possible matches"):
                                    for m in all_matches[1:4]:
                                        st.caption(f"• {to_english_city(m['canonical'])} ({m['score']}%)")
                        else:
                            st.error("❌ No good match found - please provide coordinates for a new city")
                    
                    with col2:
                        # Resolution options
                        if fuzzy.get('match_found'):
                            accept = st.checkbox(f"Accept suggestion", 
                                               value=fuzzy.get('confidence', 0) >= 90,
                                               key=f"accept_{idx}")
                            
                            if accept:
                                st.session_state.city_resolutions[city_name] = {
                                    'type': 'fuzzy_match',
                                    'canonical': fuzzy.get('suggested_canonical'),
                                    'confidence': fuzzy.get('confidence'),
                                    'english': to_english_city(fuzzy.get('suggested_canonical'))
                                }
                            else:
                                # Not accepting suggestion - show coordinate input for new city
                                st.markdown("**Add as new city:**")
                                lat = st.number_input("Latitude", key=f"lat_{idx}", value=0.0, format="%.6f")
                                lon = st.number_input("Longitude", key=f"lon_{idx}", value=0.0, format="%.6f")
                                
                                if lat != 0 and lon != 0:
                                    # Auto-detect province
                                    province, region = get_province_from_coordinates(lat, lon)
                                    if province:
                                        st.success(f"📍 Detected: {province} → {region}")
                                    else:
                                        st.warning("Could not detect province from coordinates")
                                        region = st.selectbox("Select Region", 
                                                             options=['Eastern', 'Western', 'Central', 'Northern', 'Southern'],
                                                             key=f"region_{idx}")
                                        province = None
                                    
                                    st.session_state.city_resolutions[city_name] = {
                                        'type': 'new_city',
                                        'canonical': city_name,  # Use original as canonical
                                        'latitude': lat,
                                        'longitude': lon,
                                        'province': province,
                                        'region': region
                                    }
                        else:
                            # No fuzzy match - must provide coordinates
                            st.markdown("**Add as new city:**")
                            lat = st.number_input("Latitude", key=f"lat_{idx}", value=0.0, format="%.6f")
                            lon = st.number_input("Longitude", key=f"lon_{idx}", value=0.0, format="%.6f")
                            
                            if lat != 0 and lon != 0:
                                province, region = get_province_from_coordinates(lat, lon)
                                if province:
                                    st.success(f"📍 Detected: {province} → {region}")
                                else:
                                    st.warning("Could not detect province from coordinates")
                                    region = st.selectbox("Select Region", 
                                                         options=['Eastern', 'Western', 'Central', 'Northern', 'Southern'],
                                                         key=f"region_{idx}")
                                    province = None
                                
                                st.session_state.city_resolutions[city_name] = {
                                    'type': 'new_city',
                                    'canonical': city_name,
                                    'latitude': lat,
                                    'longitude': lon,
                                    'province': province,
                                    'region': region
                                }
        
        # Check if all resolved (including ignored)
        resolved_count = len(st.session_state.city_resolutions)
        ignored_count = sum(1 for r in st.session_state.city_resolutions.values() if r.get('type') == 'ignored')
        actual_resolved = resolved_count - ignored_count
        total_unmatched = len(unmatched)
        
        st.markdown("---")
        if ignored_count > 0:
            st.markdown(f"**Resolved: {actual_resolved}/{total_unmatched}** | **Ignored: {ignored_count}**")
        else:
            st.markdown(f"**Resolved: {resolved_count}/{total_unmatched}**")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.bulk_wizard_step = 0
                st.rerun()
        
        with col2:
            # Allow proceeding when all cities are resolved OR ignored
            can_proceed = resolved_count >= total_unmatched
            if st.button("▶️ Apply Resolutions & Check Distances", type="primary", 
                        use_container_width=True, disabled=not can_proceed):
                if not can_proceed:
                    st.error("Please resolve or ignore all unmatched cities before proceeding")
                else:
                    # Apply resolutions
                    resolutions = st.session_state.city_resolutions
                    new_entries = []
                    ignored_rows = set()
                    
                    for city_name, resolution in resolutions.items():
                        if resolution['type'] == 'ignored':
                            # Track rows to skip
                            for row_num in resolution.get('rows', []):
                                ignored_rows.add(row_num)
                                
                                # 👇 FIX START: Get full row data to log Pickup/Dest
                                # parsed_rows is 0-indexed, row_num is 1-indexed
                                try:
                                    row_data = wizard_data['parsed_rows'][row_num - 1]
                                    p_val = row_data.get('pickup_raw', '')
                                    d_val = row_data.get('dest_raw', '')
                                except IndexError:
                                    p_val = "Unknown"
                                    d_val = "Unknown"
                                
                                log_exception('route_ignored', {
                                    'pickup_city': p_val,       # ✅ Explicitly log Pickup
                                    'destination_city': d_val,  # ✅ Explicitly log Destination
                                    'original_value': city_name,
                                    'action': 'Ignored by user',
                                    'row_num': row_num
                                }, immediate=False)
                                # 👆 FIX END
                                
                        elif resolution['type'] == 'fuzzy_match':
                            # Log fuzzy match
                            log_city_match(
                                original_input=city_name,
                                matched_canonical=resolution['canonical'],
                                match_type='Fuzzy Match Confirmed',
                                confidence=resolution['confidence'],
                                user=username,
                                source='bulk_upload',
                                immediate=False
                            )
                            # Add to temporary normalization
                            new_entries.append({
                                'variant': city_name,
                                'canonical': resolution['canonical'],
                                'region': get_city_region(resolution['canonical']),
                                'province': get_city_province(resolution['canonical'])
                            })
                            # Log to Append sheet
                            log_to_append_sheet(
                                variant=city_name,
                                canonical=resolution['canonical'],
                                region=get_city_region(resolution['canonical']),
                                province=get_city_province(resolution['canonical']),
                                source='fuzzy_match',
                                user=username,
                                immediate=False
                            )
                        elif resolution['type'] == 'new_city':
                            # New city
                            log_city_match(
                                original_input=city_name,
                                matched_canonical=resolution['canonical'],
                                match_type='New City',
                                confidence=None,
                                latitude=resolution.get('latitude'),
                                longitude=resolution.get('longitude'),
                                province=resolution.get('province'),
                                region=resolution.get('region'),
                                user=username,
                                source='bulk_upload',
                                immediate=False
                            )
                            new_entries.append({
                                'variant': city_name,
                                'canonical': resolution['canonical'],
                                'region': resolution.get('region'),
                                'province': resolution.get('province'),
                                'latitude': resolution.get('latitude'),
                                'longitude': resolution.get('longitude')
                            })
                            # Log to Append sheet
                            log_to_append_sheet(
                                variant=city_name,
                                canonical=resolution['canonical'],
                                region=resolution.get('region'),
                                province=resolution.get('province'),
                                latitude=resolution.get('latitude'),
                                longitude=resolution.get('longitude'),
                                source='new_city',
                                user=username,
                                immediate=False
                            )
                    
                    # Update in-memory normalization
                    update_city_normalization_pickle(new_entries)
                    
                    # ✅ FIX: Flush logs immediately
                    flush_matched_cities_to_sheet()
                    flush_append_sheet()
                    flush_error_log_to_sheet()  # Log the ignored routes we just added
                    
                    # Update parsed rows with new canonical names and mark ignored rows
                    parsed_rows = wizard_data['parsed_rows']
                    for row in parsed_rows:
                        # Check if this row should be ignored
                        row['ignored'] = row['row_num'] in ignored_rows
                        
                        if not row['ignored']:
                            if not row['pickup_ok']:
                                res = resolutions.get(row['pickup_raw'])
                                if res and res['type'] != 'ignored':
                                    row['pickup_ar'] = res['canonical']
                                    row['pickup_ok'] = True
                            if not row['dest_ok']:
                                res = resolutions.get(row['dest_raw'])
                                if res and res['type'] != 'ignored':
                                    row['dest_ar'] = res['canonical']
                                    row['dest_ok'] = True
                    
                    st.session_state.bulk_wizard_data['parsed_rows'] = parsed_rows
                    st.session_state.bulk_wizard_data['applied_resolutions'] = resolutions
                    st.session_state.bulk_wizard_data['ignored_rows'] = ignored_rows
                    
                    # Move to distance review
                    st.session_state.bulk_wizard_step = 2
                    st.rerun()
        
        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                reset_wizard()
                st.rerun()
    
    # ============================================
    # STEP 2: Distance Review
    # ============================================
    elif current_step == 2:
        st.markdown("### Step 3: Review Distances")
        
        wizard_data = st.session_state.bulk_wizard_data
        parsed_rows = wizard_data.get('parsed_rows', [])
        username = wizard_data.get('username', '')
        ignored_rows = wizard_data.get('ignored_rows', set())
        
        # Filter out ignored rows
        active_rows = [row for row in parsed_rows if not row.get('ignored', False)]
        
        if ignored_rows:
            st.info(f"ℹ️ {len(ignored_rows)} routes were ignored and will be excluded from pricing.")
        
        # Calculate distances for all routes
        if 'distance_results' not in st.session_state.bulk_wizard_data:
            st.info("Checking distances for all routes...")
            
            # FIRST: Pull in any newly-resolved distances from MatchedDistances sheet
            # This allows Google Maps API results to be used immediately
            resolved_from_sheet = get_resolved_distances_from_sheet()
            if resolved_from_sheet:
                newly_added = 0
                for item in resolved_from_sheet:
                    key = (item['pickup_ar'], item['dest_ar'])
                    rev_key = (item['dest_ar'], item['pickup_ar'])
                    # Add to DISTANCE_MATRIX temporarily (in-memory only)
                    if key not in DISTANCE_MATRIX:
                        DISTANCE_MATRIX[key] = item['distance_km']
                        DISTANCE_MATRIX[rev_key] = item['distance_km']
                        newly_added += 1
                if newly_added > 0:
                    st.success(f"📏 Auto-imported {newly_added} distances from Google Sheets API")
            
            # Get unique city pairs (only for active rows)
            city_pairs = {}
            for row in active_rows:
                pair_key = (row['pickup_ar'], row['dest_ar'])
                if pair_key not in city_pairs:
                    # Check if distance exists
                    dist, source = get_distance(row['pickup_ar'], row['dest_ar'], immediate_log=False)
                    city_pairs[pair_key] = {
                        'pickup_ar': row['pickup_ar'],
                        'dest_ar': row['dest_ar'],
                        'pickup_en': to_english_city(row['pickup_ar']),
                        'dest_en': to_english_city(row['dest_ar']),
                        'distance': dist,
                        'source': source,
                        'user_override': row.get('dist_override'),
                        'rows': []
                    }
                city_pairs[pair_key]['rows'].append(row['row_num'])
            
            # Flush missing distances to MatchedDistances sheet
            flush_ok, flush_count = flush_matched_distances_to_sheet()
            if flush_count > 0:
                st.session_state['step2_distance_flush'] = f"📏 Logged {flush_count} missing distances for Google Maps lookup"
            
            st.session_state.bulk_wizard_data['distance_results'] = city_pairs
            
            # ⚠️ DELETE any existing st.rerun() or flush calls inside this 'if' block
            # We are moving them to Fix 3 below so they run every time, not just once.

        # =======================================================
        # 👇 FIX 3 STARTS HERE (Robust Flush on Every Render)
        # =======================================================
        
        # 1. Try to flush Append Sheet (matches/new cities from Step 1)
        if 'append_sheet_pending' in st.session_state and st.session_state.append_sheet_pending:
             flush_append_sheet()
             
        # 2. Try to flush Matched Cities (fuzzy matches from Step 1)
        if 'matched_cities_pending' in st.session_state and st.session_state.matched_cities_pending:
             flush_matched_cities_to_sheet()

        # 3. Try to flush Ignored Routes errors (from Step 1)
        if 'error_log_pending' in st.session_state and st.session_state.error_log_pending:
             flush_error_log_to_sheet()

        # 4. Check/Flush Missing Distances (calculated in initialization above)
        if 'matched_distances_pending' in st.session_state and st.session_state.matched_distances_pending:
            flush_ok, flush_count = flush_matched_distances_to_sheet()
            if flush_count > 0:
                st.toast(f"✅ Sent {flush_count} missing distances to Google Sheet", icon="📏")
        
        # Force rerun ONLY if we just initialized data (to clear the loading spinner)
        # This replaces the st.rerun() that used to be inside the if block
        if 'distance_results' not in st.session_state.bulk_wizard_data:
             st.rerun()
             
        # =======================================================
        # 👆 FIX 3 ENDS HERE
        # =======================================================
        
        # Show Step 2 flush result
        if 'step2_distance_flush' in st.session_state:
            st.success(st.session_state['step2_distance_flush'])
            del st.session_state['step2_distance_flush']
        
        distance_results = st.session_state.bulk_wizard_data.get('distance_results', {})
        
        # Show any flush errors
        if 'flush_errors' in st.session_state and st.session_state.flush_errors:
            for err in st.session_state.flush_errors:
                st.warning(err)
            st.session_state.flush_errors = []  # Clear after showing
        
        # Separate missing/new distances from existing
        missing_distances = {k: v for k, v in distance_results.items() 
                           if v['distance'] == 0 or v['source'] == 'Missing'}
        existing_distances = {k: v for k, v in distance_results.items() 
                            if v['distance'] > 0 and v['source'] != 'Missing'}
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Found **{len(missing_distances)}** routes needing distance lookup, **{len(existing_distances)}** with existing distances.")
        with col2:
            if st.button("🔄 Refresh from Sheets", help="Re-check Google Sheets for newly resolved distances"):
                # Clear distance results to force re-check
                if 'distance_results' in st.session_state.bulk_wizard_data:
                    del st.session_state.bulk_wizard_data['distance_results']
                st.rerun()
        
        # Initialize distance edits state
        if 'distance_edits' not in st.session_state:
            st.session_state.distance_edits = {}
        
        # Show missing distances first (needing attention)
        if missing_distances:
            st.markdown("#### ⚠️ Routes Needing Distances")
            st.caption("These routes are missing distances. Enter values manually or wait for Google Maps API.")
            
            for pair_key, info in missing_distances.items():
                with st.expander(f"📏 {info['pickup_en']} → {info['dest_en']} ({len(info['rows'])} routes)", expanded=True):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.caption(f"Rows: {', '.join(map(str, info['rows'][:5]))}{'...' if len(info['rows']) > 5 else ''}")
                        if info['user_override']:
                            st.info(f"User provided: {info['user_override']} km")
                    
                    with col2:
                        # Distance input
                        dist_key = f"dist_{pair_key[0]}_{pair_key[1]}"
                        default_val = info['user_override'] if info['user_override'] else 0.0
                        new_dist = st.number_input(
                            "Distance (km)",
                            min_value=0.0,
                            max_value=5000.0,
                            value=float(default_val),
                            step=10.0,
                            key=dist_key
                        )
                        
                        if new_dist > 0:
                            st.session_state.distance_edits[pair_key] = new_dist
                    
                    with col3:
                        if new_dist > 0:
                            st.success("✅")
                        else:
                            st.warning("⚠️")
        
        # Show existing distances (for review/edit)
        if existing_distances:
            st.markdown("#### ✅ Routes with Distances")
            st.caption("Review and edit if needed.")
            
            # Create a dataframe for display
            dist_df = pd.DataFrame([
                {
                    'From': info['pickup_en'],
                    'To': info['dest_en'],
                    'Distance (km)': info['distance'],
                    'Source': info['source'],
                    'Routes': len(info['rows'])
                }
                for info in existing_distances.values()
            ])
            
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
            
            with st.expander("Edit existing distances"):
                for pair_key, info in existing_distances.items():
                    col1, col2 = st.columns([3, 2])
                    with col1:
                        st.text(f"{info['pickup_en']} → {info['dest_en']}")
                    with col2:
                        dist_key = f"dist_edit_{pair_key[0]}_{pair_key[1]}"
                        new_dist = st.number_input(
                            "Distance (km)",
                            min_value=0.0,
                            max_value=5000.0,
                            value=float(info['distance']),
                            step=10.0,
                            key=dist_key,
                            label_visibility="collapsed"
                        )
                        if new_dist != info['distance']:
                            st.session_state.distance_edits[pair_key] = new_dist
        
        # Navigation
        st.markdown("---")
        
        # Check if all missing distances are resolved
        all_resolved = all(
            pair_key in st.session_state.distance_edits or info['user_override']
            for pair_key, info in missing_distances.items()
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.bulk_wizard_step = 1 if wizard_data.get('unmatched_cities') else 0
                st.rerun()
        
        with col2:
            proceed_disabled = bool(missing_distances) and not all_resolved
            if st.button("▶️ Generate Pricing", type="primary", use_container_width=True,
                        disabled=proceed_disabled):
                # Log distance edits as user suggestions
                for pair_key, new_dist in st.session_state.distance_edits.items():
                    pickup_ar, dest_ar = pair_key
                    info = distance_results.get(pair_key, {})
                    old_dist = info.get('distance', 0)
                    
                    if new_dist != old_dist:
                        # Log as user suggestion to MatchedDistances
                        suggest_distance_change(
                            pickup_ar=pickup_ar,
                            dest_ar=dest_ar,
                            pickup_en=info.get('pickup_en', to_english_city(pickup_ar)),
                            dest_en=info.get('dest_en', to_english_city(dest_ar)),
                            current_distance=old_dist,
                            suggested_distance=new_dist,
                            user_name=username
                        )
                
                # Store final distances WITH their sources
                final_distances = {}
                distance_sources = {}
                for pair_key, info in distance_results.items():
                    if pair_key in st.session_state.distance_edits:
                        # User edited in Step 2
                        final_distances[pair_key] = st.session_state.distance_edits[pair_key]
                        distance_sources[pair_key] = 'User Edited'
                    elif info['user_override']:
                        # User provided in CSV
                        final_distances[pair_key] = info['user_override']
                        distance_sources[pair_key] = 'CSV Provided'
                    else:
                        # From matrix/historical - store distance but mark source so we know not to override
                        final_distances[pair_key] = info['distance']
                        distance_sources[pair_key] = info['source']  # 'Matrix', 'Historical', etc.
                
                st.session_state.bulk_wizard_data['final_distances'] = final_distances
                st.session_state.bulk_wizard_data['distance_sources'] = distance_sources
                st.session_state.bulk_wizard_step = 3
                st.rerun()
        
        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                reset_wizard()
                st.rerun()
    
    # ============================================
    # STEP 3: Final Pricing
    # ============================================
    elif current_step == 3:
        st.markdown("### Step 4: Pricing Results")
        
        wizard_data = st.session_state.bulk_wizard_data
        parsed_rows = wizard_data.get('parsed_rows', [])
        selected_vehicles = wizard_data.get('selected_vehicles', ['Flatbed Trailer'])
        final_distances = wizard_data.get('final_distances', {})
        username = wizard_data.get('username', '')
        ignored_rows = wizard_data.get('ignored_rows', set())
        
        # Filter out ignored rows
        active_rows = [row for row in parsed_rows if not row.get('ignored', False)]
        
        if ignored_rows:
            st.info(f"ℹ️ {len(ignored_rows)} routes were ignored and excluded from pricing.")
        
        if not active_rows:
            st.warning("⚠️ All routes were ignored. No pricing to generate.")
            if st.button("🔄 Start New Bulk Lookup", use_container_width=True):
                reset_wizard()
                st.rerun()
        else:
            # Generate pricing
            if 'bulk_results' not in st.session_state or st.session_state.get('bulk_results_stale', True):
                results = []
                
                progress = st.progress(0)
                status = st.empty()
                
                for i, row in enumerate(active_rows):
                    pickup_ar = row['pickup_ar']
                    dest_ar = row['dest_ar']
                    pair_key = (pickup_ar, dest_ar)
                    
                    # Only use dist_override for user-edited or CSV-provided distances
                    # Let lookup_route_stats find Matrix/Historical distances naturally
                    distance_sources = wizard_data.get('distance_sources', {})
                    source = distance_sources.get(pair_key, '')
                    
                    if source in ['User Edited', 'CSV Provided']:
                        dist_override = final_distances.get(pair_key, row.get('dist_override'))
                    else:
                        # Let the function look up the distance naturally to get correct source
                        dist_override = None
                    
                    # Price for each selected vehicle
                    for v_en in selected_vehicles:
                        v_ar = to_arabic_vehicle(v_en)
                        
                        result = lookup_route_stats(
                            pickup_ar, dest_ar, v_ar,
                            dist_override=dist_override
                        )
                        result['CSV_Row'] = row['row_num']
                        results.append(result)
                    
                    progress.progress((i + 1) / len(active_rows))
                    status.text(f"Pricing {i + 1}/{len(active_rows)}: {row['pickup_raw']} → {row['dest_raw']}")
                
                status.text("✅ Complete!")
                st.session_state.bulk_results = pd.DataFrame(results)
                st.session_state.bulk_results_stale = False
                
                # Flush logs
                flush_error_log_to_sheet()
                flush_matched_distances_to_sheet()
            
            # Display results
            res_df = st.session_state.bulk_results
            
            st.success(f"Generated pricing for **{len(res_df)}** route-vehicle combinations")
            
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            # Save section
            st.markdown("---")
            st.subheader("💾 Save Results")
            
            rfq_sheet_url = st.secrets.get('RFQ_url', '')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "📥 Download CSV",
                    res_df.to_csv(index=False),
                    "pricing_results.csv",
                    "text/csv",
                    type="primary",
                    use_container_width=True
                )
            
            with col2:
                if rfq_sheet_url:
                    default_sheet_name = f"Pricing_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    sheet_name = st.text_input("Sheet Name", value=default_sheet_name, key='result_sheet_name')
                    
                    if st.button("☁️ Upload to Google Sheet", use_container_width=True):
                        progress = st.progress(0)
                        status = st.empty()
                        
                        def update_progress(done, total, batch):
                            progress.progress(done / total)
                            status.text(f"Uploading {done}/{total}...")
                        
                        ok, msg = upload_to_rfq_sheet(res_df, sheet_name, progress_callback=update_progress)
                        progress.progress(1.0)
                        status.empty()
                        
                        if ok:
                            st.success(msg)
                            st.link_button("🔗 Open Sheet", rfq_sheet_url)
                        else:
                            st.error(msg)
            
            # Navigation
            st.markdown("---")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("⬅️ Back to Distances", use_container_width=True):
                    st.session_state.bulk_results_stale = True
                    st.session_state.bulk_wizard_step = 2
                    st.rerun()
            
            with col2:
                if st.button("🔄 Start New Bulk Lookup", type="secondary", use_container_width=True):
                    reset_wizard()
                    st.rerun()
    # ============================================
    # MASTER GRID GENERATOR
    # Generates pricing for all city-to-city combinations
    # Uploads directly to Google Sheet for reference pricing
    # ============================================
    st.markdown("---")
    with st.expander("⚡ Admin: Generate Master Grid (All Cities)"):
        st.warning("⚠️ This will generate pricing for ALL city combinations (~2,500+ routes). It may take a few minutes.")
        st.caption(f"Results will be uploaded to: {BULK_PRICING_SHEET_URL}")
        
        batch_size = st.slider("Batch size", min_value=100, max_value=500, value=250, step=50, 
                               help="Number of routes to generate and upload at a time")
        
        # Check if there's a previous incomplete run
        has_incomplete = 'master_grid_progress' in st.session_state and st.session_state.master_grid_progress.get('incomplete', False)
        
        if has_incomplete:
            prog = st.session_state.master_grid_progress
            
            # Show error message if there was one
            if prog.get('error_message'):
                st.error(f"❌ {prog['error_message']}")
            
            st.warning(f"⚠️ Previous run incomplete: {prog['rows_written']:,}/{prog['total_routes']:,} routes uploaded. You can resume from here.")
            
            col1, col2 = st.columns(2)
            with col1:
                resume_clicked = st.button("🔄 Resume Upload", type="primary")
            with col2:
                if st.button("🗑️ Start Fresh"):
                    del st.session_state.master_grid_progress
                    if 'master_grid_df' in st.session_state:
                        del st.session_state.master_grid_df
                    st.rerun()
        else:
            resume_clicked = False
        
        start_fresh_clicked = st.button("🚀 Run & Upload Master Grid") if not has_incomplete else False
        
        if start_fresh_clicked or resume_clicked:
            import time
            
            # Get worksheet connection first
            wks = get_bulk_sheet()
            if not wks:
                st.error("❌ Google Cloud credentials not found or sheet access denied.")
            else:
                # Build list of all route combinations (excluding same-city)
                combos = [(p, d) for p, d in itertools.product(all_canonicals, all_canonicals) if p != d]
                total_routes = len(combos)
                
                # Ensure sheet has enough rows (routes + header + timestamp + buffer)
                ensure_sheet_rows(wks, total_routes + 10)
                
                # Determine starting point
                if resume_clicked and has_incomplete:
                    # Resume from where we left off
                    prog = st.session_state.master_grid_progress
                    start_index = prog['next_index']
                    rows_written = prog['rows_written']
                    batch_num = prog['batch_num']
                    all_results = prog.get('all_results', [])
                    st.info(f"📊 Resuming from route {start_index:,} ({rows_written:,} already uploaded)...")
                else:
                    # Fresh start - clear sheet and write headers
                    try:
                        wks.clear()
                        headers = ['From', 'To', 'Distance', 'Buy_Price', 'Rec_Sell', 'Ref_Sell', 
                                   'Ref_Sell_Src', 'Rental_Cost', 'Margin', 'Model', 'Confidence', 'Recent_N']
                        wks.update('A1', [headers])
                    except Exception as e:
                        st.error(f"❌ Failed to initialize sheet: {str(e)}")
                        st.stop()
                    
                    start_index = 0
                    rows_written = 0
                    batch_num = 0
                    all_results = []
                    st.info(f"📊 Generating and uploading {total_routes:,} routes in batches of {batch_size}...")
                
                progress_bar = st.progress(rows_written / total_routes if total_routes > 0 else 0)
                status_text = st.empty()
                
                # Process in batches: generate → write → repeat
                for i in range(start_index, total_routes, batch_size):
                    batch_combos = combos[i:i + batch_size]
                    batch_num += 1
                    
                    # Generate prices for this batch
                    status_text.text(f"Batch {batch_num}: Generating {len(batch_combos)} routes...")
                    batch_results = []
                    for p_ar, d_ar in batch_combos:
                        batch_results.append(lookup_route_stats(p_ar, d_ar, DEFAULT_VEHICLE_AR, check_history=True))
                    
                    all_results.extend(batch_results)
                    
                    # Check if we need to flush error logs and matched distances (every 500 errors)
                    pending_errors = len(st.session_state.get('error_log_pending', []))
                    pending_distances = len(st.session_state.get('matched_distances_pending', []))
                    if pending_errors >= 500:
                        flush_error_log_to_sheet(batch_size=500)
                    if pending_distances >= 100:
                        flush_matched_distances_to_sheet()
                    
                    # Write this batch to sheet
                    status_text.text(f"Batch {batch_num}: Writing {len(batch_results)} rows to sheet...")
                    try:
                        start_row = rows_written + 2  # +2 for header and 1-indexing
                        batch_df = pd.DataFrame(batch_results)
                        data_rows = batch_df.astype(str).values.tolist()
                        wks.update(f'A{start_row}', data_rows)
                        rows_written += len(batch_results)
                    except Exception as e:
                        # Save state for resume
                        st.session_state.master_grid_progress = {
                            'incomplete': True,
                            'next_index': i + batch_size,
                            'rows_written': rows_written,
                            'batch_num': batch_num,
                            'total_routes': total_routes,
                            'all_results': all_results,
                            'error_message': f"Failed at batch {batch_num}: {str(e)}"
                        }
                        # Flush any pending errors and matched distances before rerun
                        flush_error_log_to_sheet(batch_size=500)
                        flush_matched_distances_to_sheet()
                        st.rerun()
                    
                    # Update progress
                    progress_bar.progress(min(1.0, (i + len(batch_combos)) / total_routes))
                    status_text.text(f"✓ Batch {batch_num} complete | {rows_written:,}/{total_routes:,} routes uploaded")
                    
                    # Small delay to avoid rate limiting
                    if i + batch_size < total_routes:
                        time.sleep(0.3)
                
                # If we get here, loop completed successfully (errors cause st.rerun())
                # Add timestamp
                try:
                    wks.update(f'A{rows_written + 2}', [[f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]])
                except:
                    pass
                
                progress_bar.progress(1.0)
                st.success(f"✅ Successfully uploaded {rows_written:,} routes in {batch_num} batches!")
                status_text.empty()
                
                # Clear incomplete state
                if 'master_grid_progress' in st.session_state:
                    del st.session_state.master_grid_progress
                
                # Flush any remaining pending error logs
                flushed_ok, flushed_count = flush_error_log_to_sheet(batch_size=500)
                if flushed_count > 0:
                    st.caption(f"📝 Logged {flushed_count} exceptions to error sheet")
                
                # Flush any remaining pending matched distances
                dist_ok, dist_count = flush_matched_distances_to_sheet()
                if dist_count > 0:
                    st.caption(f"📏 Logged {dist_count} missing distances for Google Maps lookup")
                
                # Store for download
                st.session_state.master_grid_df = pd.DataFrame(all_results)
        
        # Allow download of master grid if it was generated
        if 'master_grid_df' in st.session_state:
            st.download_button(
                "📥 Download Master Grid CSV", 
                st.session_state.master_grid_df.to_csv(index=False), 
                "master_grid.csv", 
                "text/csv"
            )
    
    # ============================================
    # DISTANCE UPDATE FROM GOOGLE SHEETS
    # Pull resolved distances from MatchedDistances sheet
    # ============================================
    st.markdown("---")
    with st.expander("📏 Admin: Update Distances from Google Sheets"):
        # Inject CSS to strictly force centering on Metric elements
        st.markdown("""
            <style>
            /* Center the main metric container */
            div[data-testid="stMetric"] {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                width: 100%;
            }
            /* Center the label (top text) */
            div[data-testid="stMetricLabel"] {
                justify-content: center;
                width: 100%;
                margin: auto;
            }
            /* Center the value (big number) */
            div[data-testid="stMetricValue"] {
                justify-content: center;
                width: 100%;
                margin: auto;
            }
            /* Center the delta (arrow text) if present */
            div[data-testid="stMetricDelta"] {
                justify-content: center;
                width: 100%;
                margin: auto;
            }
            </style>
            """, unsafe_allow_html=True)

        # 1. Dashboard Metrics
        resolved = get_resolved_distances_from_sheet()
        failed = get_failed_distances_from_sheet()
        
        st.markdown("<h3 style='text-align: center;'>📊 Status Dashboard</h3>", unsafe_allow_html=True)
        
        # Use spacers to center the 3 metrics in the middle of the screen
        # Layout: [spacer, metric, metric, metric, spacer]
        _, m1, m2, m3, _ = st.columns([1, 2, 2, 2, 1])
        
        m1.metric("Ready to Import", len(resolved), help="Distances found in Sheet ready to be added")
        m2.metric("Failed Lookups", len(failed), delta_color="inverse" if len(failed) > 0 else "off", help="Routes Google couldn't find")
        m3.metric("Current Pickle Size", f"{len(DISTANCE_MATRIX):,}", help="Total distances currently in memory")
        
        # 2. Detailed Data Views (Tabs)
        if resolved or failed:
            # DETAILS: Left-aligned
            st.markdown("<h4 style='text-align: left;'>📋 Details</h4>", unsafe_allow_html=True)
            t1, t2 = st.tabs(["✅ Ready to Import", "⚠️ Failed Lookups"])
            
            with t1:
                if resolved:
                    # Description: Left-aligned
                    st.markdown("<p style='text-align: left; color: gray;'>These distances will be added to your local database.</p>", unsafe_allow_html=True)
                    preview_df = pd.DataFrame([
                        {'From': to_english_city(r['pickup_ar']), 
                            'To': to_english_city(r['dest_ar']), 
                            'Distance': f"{r['distance_km']} km",
                            'Source': '✏️ User' if r.get('is_suggestion') else '🌐 API',
                            'Row': r['row']}
                        for r in resolved
                    ])
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No pending distances to import.")

            with t2:
                if failed:
                    # Description: Left-aligned
                    st.markdown("<p style='text-align: left; color: orange;'>These routes failed API lookup. Please manually enter distances in Column G of the Sheet.</p>", unsafe_allow_html=True)
                    failed_df = pd.DataFrame([
                        {'Row': f['row'],
                            'From': f['pickup_en'] or to_english_city(f['pickup_ar']),
                            'To': f['dest_en'] or to_english_city(f['dest_ar']),
                            'Error': f['error_value']}
                        for f in failed
                    ])
                    st.dataframe(failed_df, use_container_width=True, hide_index=True)
                else:
                    st.success("No failed lookups! All clean.")

        st.markdown("---")
        
        # 3. Actions Area (Save & Sync)
        st.markdown("<h5 style='text-align: center;'>💾 Save & Sync</h5>", unsafe_allow_html=True)
        
        # Center buttons using spacer columns
        _, c1, c2, c3, _ = st.columns([1, 2, 2, 2, 1])
        
        with c1:
            refresh_clicked = st.button("🔍 Refresh Sheet Data", use_container_width=True)

        with c2:
            import_clicked = st.button("🔄 Import to Pickle", type="primary", disabled=len(resolved)==0, use_container_width=True)
            
        with c3:
            # Read the current file from memory/disk for download
            pkl_path = os.path.join(APP_DIR, 'model_export', 'distance_matrix.pkl')
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    pkl_data = f.read()
                    
                st.download_button(
                    label="📥 Download .pkl",
                    data=pkl_data,
                    file_name="distance_matrix.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                    type="secondary"
                )

        if refresh_clicked:
            st.rerun()

        # Import Logic (Preserved)
        if import_clicked:
            with st.spinner("Importing distances..."):
                success, count, message = update_distance_pickle_from_sheet()
                if success:
                    st.success(message)
                    if count > 0:
                        st.balloons()
                        st.cache_resource.clear()  # Clear cached data to reload
                    
                    # Re-check for failed lookups immediately AFTER import
                    failed_now = get_failed_distances_from_sheet()
                    if failed_now:
                        st.error(f"⚠️ {len(failed_now)} routes still have failed Google Maps lookups.")
                else:
                    st.error(message)

# --- TAB 3: MAP EXPLORER (OSM ROUTING) ---
# --- TAB 3: MAP EXPLORER (KEY ROTATION FIX) ---
with tab3:
    st.subheader("🗺️ Live Distance Calculator (OSM)")

    # 1. HELPER: Key Rotation to force-refresh inputs
    def force_update_pin(prefix, lat, lon, name=None):
        """
        Updates variable AND increments a counter.
        The counter change forces the Input Widgets to be destroyed and re-created,
        guaranteeing they display the new value.
        """
        # Update the Source of Truth
        st.session_state[f'{prefix}_lat'] = lat
        st.session_state[f'{prefix}_lon'] = lon
        if name: st.session_state[f'{prefix}_name'] = name
        
        # Increment the "Refresh ID" to force new widgets
        st.session_state['refresh_id'] += 1
        
        # Reset Route & Map View
        st.session_state.route_geom = None
        st.session_state.map_center = [lat, lon]
        st.session_state.map_zoom = 12

    # 2. Initialize Defaults
    if 'refresh_id' not in st.session_state: st.session_state.refresh_id = 0
    
    if 'p1_lat' not in st.session_state:
        st.session_state.p1_lat = 24.7136
        st.session_state.p1_lon = 46.6753
        st.session_state.p1_name = "Riyadh"
    
    if 'p2_lat' not in st.session_state:
        st.session_state.p2_lat = 21.5433
        st.session_state.p2_lon = 39.1728
        st.session_state.p2_name = "Jeddah"

    if 'map_center' not in st.session_state: st.session_state.map_center = [24.7136, 46.6753]
    if 'map_zoom' not in st.session_state: st.session_state.map_zoom = 6
    if 'route_geom' not in st.session_state: st.session_state.route_geom = None
    if 'dist_display' not in st.session_state: st.session_state.dist_display = 0

    col_ctrl, col_map = st.columns([1, 2])
    
    with col_ctrl:
        st.markdown("### 📍 Location Adjustment")
        
        # --- ORIGIN CONTROLS ---
        st.markdown("#### 🟢 Origin")
        
        # Search Bar
        c_search1, c_btn1 = st.columns([3, 1])
        with c_search1:
            search_1 = st.text_input("Search Origin", label_visibility="collapsed", placeholder="City or District...", key="s1")
        with c_btn1:
            if st.button("Find", key="btn_find_p1"):
                with st.spinner("..."):
                    res = search_osm(search_1)
                    if res:
                        force_update_pin('p1', res[0], res[1], res[2])
                        st.rerun()
                    else: st.error("❌")
        
        # MANUAL INPUTS (With Dynamic Keys)
        # We append 'refresh_id' to the key. This forces a fresh widget on every update.
        c1, c2 = st.columns(2)
        rid = st.session_state.refresh_id
        
        new_p1_lat = c1.number_input("Lat", value=st.session_state.p1_lat, format="%.5f", key=f"p1_lat_{rid}")
        new_p1_lon = c2.number_input("Lon", value=st.session_state.p1_lon, format="%.5f", key=f"p1_lon_{rid}")
        
        # Detect Manual Typing (Update state without incrementing counter)
        if new_p1_lat != st.session_state.p1_lat or new_p1_lon != st.session_state.p1_lon:
            st.session_state.p1_lat = new_p1_lat
            st.session_state.p1_lon = new_p1_lon
            st.session_state.route_geom = None
            st.rerun()

        st.markdown("---")

        # --- DESTINATION CONTROLS ---
        st.markdown("#### 🔴 Destination")
        
        # Search Bar
        c_search2, c_btn2 = st.columns([3, 1])
        with c_search2:
            search_2 = st.text_input("Search Destination", label_visibility="collapsed", placeholder="City or District...", key="s2")
        with c_btn2:
            if st.button("Find", key="btn_find_p2"):
                with st.spinner("..."):
                    res = search_osm(search_2)
                    if res:
                        force_update_pin('p2', res[0], res[1], res[2])
                        st.rerun()
                    else: st.error("❌")

        # MANUAL INPUTS (With Dynamic Keys)
        c3, c4 = st.columns(2)
        new_p2_lat = c3.number_input("Lat", value=st.session_state.p2_lat, format="%.5f", key=f"p2_lat_{rid}")
        new_p2_lon = c4.number_input("Lon", value=st.session_state.p2_lon, format="%.5f", key=f"p2_lon_{rid}")

        # Detect Manual Typing
        if new_p2_lat != st.session_state.p2_lat or new_p2_lon != st.session_state.p2_lon:
            st.session_state.p2_lat = new_p2_lat
            st.session_state.p2_lon = new_p2_lon
            st.session_state.route_geom = None
            st.rerun()

        st.markdown("---")
        
        # --- CALCULATION ---
        if st.session_state.p1_lat and st.session_state.p2_lat:
            if st.session_state.route_geom is None:
                with st.spinner("Calculating route..."):
                    dist_km, geometry = get_osrm_route(
                        [st.session_state.p1_lat, st.session_state.p1_lon],
                        [st.session_state.p2_lat, st.session_state.p2_lon]
                    )
                    st.session_state.dist_display = dist_km
                    st.session_state.route_geom = geometry
            
            if st.session_state.get('dist_display'):
                st.metric("🚗 Driving Distance", f"{st.session_state.dist_display:,.1f} km")
                st.caption("Route: OSRM (Road Network)")
            else:
                st.warning("No drivable route found.")

    with col_map:
        click_mode = st.radio("🖱️ Map Click Action:", ["Do Nothing", "Set Origin (Green)", "Set Destination (Red)"], horizontal=True)
        
        m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, tiles="CartoDB positron")
        
        folium.Marker([st.session_state.p1_lat, st.session_state.p1_lon], tooltip="Origin", icon=folium.Icon(color="green", icon="play")).add_to(m)
        folium.Marker([st.session_state.p2_lat, st.session_state.p2_lon], tooltip="Destination", icon=folium.Icon(color="red", icon="stop")).add_to(m)

        if st.session_state.route_geom:
            folium.PolyLine(st.session_state.route_geom, color="blue", weight=4, opacity=0.7).add_to(m)
            m.fit_bounds([[st.session_state.p1_lat, st.session_state.p1_lon], [st.session_state.p2_lat, st.session_state.p2_lon]], padding=(50, 50))

        st_data = st_folium(m, width="100%", height=600, returned_objects=["last_clicked"])
        
        # --- ROBUST CLICK HANDLER ---
        if st_data and st_data.get('last_clicked'):
            lat = st_data['last_clicked']['lat']
            lon = st_data['last_clicked']['lng']
            
            changed = False
            
            if "Set Origin" in click_mode:
                if abs(lat - st.session_state.p1_lat) > 0.00001:
                    force_update_pin('p1', lat, lon)
                    changed = True

            elif "Set Destination" in click_mode:
                if abs(lat - st.session_state.p2_lat) > 0.00001:
                    force_update_pin('p2', lat, lon)
                    changed = True
            
            if changed:
                st.rerun()
                
# ERROR LOG SECTION (Restored to exact snippet)
error_log_csv = get_error_log_csv()
if error_log_csv:
    with st.expander("⚠️ Exception Log", expanded=False):
        st.caption(f"{len(st.session_state.error_log)} exceptions logged this session")
        st.dataframe(pd.DataFrame(st.session_state.error_log), use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download Error Log", error_log_csv, "pricing_error_log.csv", "text/csv")
        with col2:
            if st.button("🗑️ Clear Log"):
                clear_error_log()
                st.rerun()

st.caption("Freight Pricing Tool | Buy Price rounded to 100 | Sell Price rounded to 50 | All prices in SAR")
