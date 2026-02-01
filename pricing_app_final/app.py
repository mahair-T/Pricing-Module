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

# City coordinates for user-added cities (to skip Google Maps formula)
# Using a function to access session_state (initialized on first access)
def get_city_coordinates():
    """Get CITY_COORDINATES from session state (persists across reruns)."""
    try:
        if 'CITY_COORDINATES' not in st.session_state:
            st.session_state.CITY_COORDINATES = {}
        return st.session_state.CITY_COORDINATES
    except:
        return {}  # Return empty dict during startup (before session_state exists)

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

def get_geocode_lookup_sheet():
    """
    Get or create the GeocodeLookup sheet for automatic Google Maps geocoding.
    Uses GOOGLEMAPS_LATLONG formula to resolve unmatched city names.
    
    Columns:
    A: Input (city name to geocode)
    B: Latitude (formula result)
    C: Longitude (formula result)  
    D: Matched Name (formula result - what Google thinks this city is)
    E: Status (Pending/Resolved/Error)
    F: Timestamp
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
            worksheet = spreadsheet.worksheet('GeocodeLookup')
        except:
            # Create sheet with headers
            worksheet = spreadsheet.add_worksheet(title='GeocodeLookup', rows=500, cols=6)
            headers = ['Input', 'Latitude', 'Longitude', 'Matched_Name', 'Status', 'Timestamp']
            worksheet.update('A1:F1', [headers])
        
        return worksheet
    except Exception as e:
        return None

def write_cities_for_geocoding(city_names):
    """
    Write unmatched city names to GeocodeLookup sheet with GOOGLEMAPS_LATLONG formula.
    
    Args:
        city_names: List of city names to geocode
        
    Returns:
        (success, count_written, start_row) tuple
    """
    if not city_names:
        return True, 0, 0
    
    try:
        worksheet = get_geocode_lookup_sheet()
        if not worksheet:
            return False, 0, 0
        
        # Get existing data to find next row and check for duplicates
        existing = worksheet.get_all_values()
        existing_inputs = set()
        for row in existing[1:]:  # Skip header
            if len(row) > 0:
                existing_inputs.add(row[0].strip().lower())
        
        next_row = len(existing) + 1
        start_row = next_row
        rows_to_add = []
        
        for city in city_names:
            # Skip if already in sheet
            if city.strip().lower() in existing_inputs:
                continue
            
            # Build row with GOOGLEMAPS_LATLONG formula
            # The formula returns [[lat, lng, address]] so we need INDEX to extract each part
            formula_lat = f'=IFERROR(INDEX(GOOGLEMAPS_LATLONG(A{next_row}), 1, 1), "")'
            formula_lon = f'=IFERROR(INDEX(GOOGLEMAPS_LATLONG(A{next_row}), 1, 2), "")'
            formula_name = f'=IFERROR(INDEX(GOOGLEMAPS_LATLONG(A{next_row}), 1, 3), "")'
            
            row_data = [
                city,                                           # A: Input
                formula_lat,                                    # B: Latitude
                formula_lon,                                    # C: Longitude
                formula_name,                                   # D: Matched Name
                'Pending',                                      # E: Status
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')   # F: Timestamp
            ]
            rows_to_add.append(row_data)
            existing_inputs.add(city.strip().lower())
            next_row += 1
        
        if rows_to_add:
            # Ensure sheet has enough rows
            current_row_count = worksheet.row_count
            required_rows = start_row + len(rows_to_add) + 10
            
            if required_rows > current_row_count:
                worksheet.add_rows(max(len(rows_to_add) + 50, 100))
            
            # Write all rows at once
            end_row = start_row + len(rows_to_add) - 1
            worksheet.update(f'A{start_row}:F{end_row}', rows_to_add, value_input_option='USER_ENTERED')
        
        return True, len(rows_to_add), start_row
    
    except Exception as e:
        if 'flush_errors' not in st.session_state:
            st.session_state.flush_errors = []
        st.session_state.flush_errors.append(f"Geocode write error: {str(e)}")
        return False, 0, 0

def read_geocode_results(city_names):
    """
    Read geocoding results from GeocodeLookup sheet for specified cities.
    
    Args:
        city_names: List of city names to get results for
        
    Returns:
        Dict mapping city_name -> {latitude, longitude, matched_name, success}
    """
    results = {}
    
    if not city_names:
        return results
    
    try:
        worksheet = get_geocode_lookup_sheet()
        if not worksheet:
            return results
        
        # Read all data
        all_data = worksheet.get_all_values()
        
        # Build lookup for requested cities (case-insensitive)
        city_lookup = {c.strip().lower(): c for c in city_names}
        
        for row in all_data[1:]:  # Skip header
            if len(row) >= 4:
                input_city = row[0].strip()
                input_lower = input_city.lower()
                
                if input_lower in city_lookup:
                    original_name = city_lookup[input_lower]
                    lat_str = row[1] if len(row) > 1 else ''
                    lon_str = row[2] if len(row) > 2 else ''
                    matched_name = row[3] if len(row) > 3 else ''
                    
                    # Check if we have valid results
                    try:
                        lat = float(lat_str) if lat_str else None
                        lon = float(lon_str) if lon_str else None
                        
                        if lat and lon and matched_name:
                            results[original_name] = {
                                'latitude': lat,
                                'longitude': lon,
                                'matched_name': matched_name.strip(),
                                'success': True
                            }
                        else:
                            results[original_name] = {
                                'latitude': None,
                                'longitude': None,
                                'matched_name': None,
                                'success': False
                            }
                    except (ValueError, TypeError):
                        results[original_name] = {
                            'latitude': None,
                            'longitude': None,
                            'matched_name': None,
                            'success': False
                        }
        
        return results
    
    except Exception as e:
        return results

def clear_geocode_lookup_sheet():
    """Clear all data from GeocodeLookup sheet except headers."""
    try:
        worksheet = get_geocode_lookup_sheet()
        if worksheet:
            # Get row count, clear data rows but keep header
            row_count = worksheet.row_count
            if row_count > 1:
                worksheet.delete_rows(2, row_count)
            return True
    except Exception as e:
        pass
    return False

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
        
        # Store coordinates for skipping Google Maps formula
        latitude = entry.get('latitude')
        longitude = entry.get('longitude')
        if latitude and longitude and latitude != 0 and longitude != 0:
            city_coords = get_city_coordinates()
            city_coords[canonical] = (latitude, longitude)
            if variant != canonical:
                city_coords[variant] = (latitude, longitude)
        
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
            worksheet = spreadsheet.add_worksheet(title='Bulk Adjustments', rows=1000, cols=16)
            headers = [
                'Timestamp', 'Row_Number', 'Pickup_City', 'Destination_City', 
                'Pickup_EN', 'Destination_EN', 'Freight_Segment', 'User_Distance', 
                'Pickup_Lat', 'Pickup_Lon', 'Dropoff_Lat', 'Dropoff_Lon',
                'Detected_Pickup_Province', 'Detected_Dropoff_Province',
                'Detected_Pickup_Region', 'Detected_Dropoff_Region'
            ]
            worksheet.update('A1:P1', [headers])
        
        return worksheet
    except Exception as e:
        return None

def log_bulk_adjustment(row_num, pickup_city, dest_city, pickup_en, dest_en, 
                        freight_segment='Domestic', user_distance=None, 
                        pickup_lat=None, pickup_lon=None, 
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
        freight_segment,
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

# ============================================
# TRUCK TYPES (Domestic vs Port Pricing)
# ============================================
TRUCK_TYPES = ['Domestic', 'Port Direct', 'Port Roundtrip']
DEFAULT_TRUCK_TYPE = 'Domestic'

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
# INDEX + SHRINKAGE MODEL (90d Half-Life)
# Used for lanes WITH some historical data
# Combines market index trends with shrinkage estimation
# Uses exponential decay weighting with 90-day half-life
# ============================================
class IndexShrinkagePredictor:
    """
    Index + Shrinkage model for lanes with some historical data.
    
    Two-component prediction:
    1. Index: Uses lane-specific multiplier × current market index
       - Captures how this lane moves relative to market
       - Uses exponential decay weighting (90d half-life)
    2. Shrinkage: Bayesian estimate that shrinks lane mean toward city prior
       - Handles sparse data by borrowing strength from similar lanes
    
    Final prediction = average of Index and Shrinkage components
    """
    
    def __init__(self, model_artifacts):
        m = model_artifacts
        self.current_index = m['current_index']      # Current market index value
        self.lane_multipliers = m['lane_multipliers'] # Lane-specific multipliers
        self.global_mean = m['global_mean']          # Global mean CPK (fallback)
        # Handle both old ('k_prior_strength') and new ('k') model formats
        self.k = m.get('k', m.get('k_prior_strength', 10))  # Shrinkage strength parameter
        self.pickup_priors = m['pickup_priors']      # City-level priors for pickup
        self.dest_priors = m['dest_priors']          # City-level priors for destination
        self.lane_stats = m['lane_stats']            # Historical lane statistics
        self.city_to_region = m.get('city_to_region', {})
        self.regional_cpk = {tuple(k.split('|')): v for k, v in m.get('regional_cpk', {}).items()} if isinstance(list(m.get('regional_cpk', {}).keys())[0] if m.get('regional_cpk') else '', str) else m.get('regional_cpk', {})
        # Get model config info if available
        config = m.get('config', {})
        self.model_date = config.get('training_date', m.get('model_date', 'Unknown'))
        self.half_life_days = config.get('half_life_days', 90)
    
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
            # Use effective sample size ('lane_n') for shrinkage calculation
            # This accounts for exponential decay weighting in the 90d HL model
            lane_n = stats['lane_n']
            lam = lane_n / (lane_n + self.k)
            shrink_pred = lam * lane_mean + (1 - lam) * city_prior
            # Use raw trip count ('raw_n') for confidence if available, else fall back to lane_n
            raw_n = stats.get('raw_n', lane_n)
        else:
            shrink_pred = city_prior
            raw_n = 0
            
        # Combine
        if idx_pred is not None:
            predicted_cpk = (idx_pred + shrink_pred) / 2
            method = 'Index + Shrinkage'
            if stats:
                confidence = 'High' if raw_n >= 20 else 'Medium' if raw_n >= 5 else 'Low'
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
# SPATIAL MODEL (R150 P50 IDW - New Lane Model)
# Used for completely NEW lanes with NO historical data
# Uses inverse-distance weighted CPK from nearby cities
# Falls back to Province P50 percentile
# ============================================
class SpatialPredictor:
    """
    Spatial R150 P50 IDW model for completely new lanes.
    
    For lanes with zero historical data, we estimate price by:
    1. Spatial IDW: Find nearby cities within 150km radius, compute
       inverse-distance weighted average of their CPK values
    2. Province P50: If not enough neighbors, use province pair's 50th percentile
    3. Global Mean: Last resort fallback
    
    This provides a reasonable estimate even for never-seen routes by
    leveraging geographic proximity patterns in freight pricing.
    """
    
    def __init__(self, model_artifacts):
        m = model_artifacts
        self.config = m.get('config', {})
        self.radius_km = self.config.get('radius_km', 150)
        self.percentile = self.config.get('percentile', 50)
        self.min_neighbors = self.config.get('min_neighbors', 3)
        
        # Neighbor lookup (city -> [(neighbor_city, distance), ...])
        self.city_neighbors = m.get('city_neighbors', {})
        
        # City-level CPK statistics
        self.city_outbound_cpk = m.get('city_outbound_cpk', {})
        self.city_inbound_cpk = m.get('city_inbound_cpk', {})
        
        # Cascade V1: Region × Distance fallback
        self.region_distance_cpk = m.get('region_distance_cpk', {})
        self.city_to_region = m.get('city_to_region', {})
        self.distance_quantiles = m.get('distance_quantiles', {})
        
        # Cascade V1: Distance Band fallback
        self.distance_band_cpk = m.get('distance_band_cpk', {})
        
        # Province fallback
        self.province_pair_cpk = m.get('province_pair_cpk', {})
        self.city_to_province = m.get('city_to_province', {})
        
        # Global fallback
        self.global_mean = m.get('global_mean', 2.0)
        
        self.model_date = self.config.get('training_date', 'Unknown')
    
    def _get_nearby_cities(self, city, radius_km):
        """Get cities within radius."""
        if city not in self.city_neighbors:
            return []
        return [(c, d) for c, d in self.city_neighbors[city] if d <= radius_km]
    
    def predict(self, pickup_city, dest_city, distance_km=None,
                pickup_region_override=None, dest_region_override=None):
        """
        Predict CPK for a lane using spatial neighbor interpolation.
        
        Args:
            pickup_city: Pickup city (Arabic canonical or English)
            dest_city: Destination city (Arabic canonical or English)
            distance_km: Distance in km (optional, used to calculate total cost)
            pickup_region_override: Override region for pickup (from coordinates) - unused but kept for API compatibility
            dest_region_override: Override region for destination (from coordinates) - unused but kept for API compatibility
        """
        # Get canonical names
        p_can = CITY_TO_CANONICAL.get(pickup_city, pickup_city)
        d_can = CITY_TO_CANONICAL.get(dest_city, dest_city)
        
        # Try spatial IDW first
        pickup_neighbors = self._get_nearby_cities(p_can, self.radius_km)
        pickup_data = []
        for neighbor, dist in pickup_neighbors:
            if neighbor in self.city_outbound_cpk:
                cpk_info = self.city_outbound_cpk[neighbor]
                # Handle both dict and simple float formats
                cpk = cpk_info.get('outbound_cpk_mean', cpk_info) if isinstance(cpk_info, dict) else cpk_info
                pickup_data.append({'cpk': cpk, 'dist': dist})
        
        dest_neighbors = self._get_nearby_cities(d_can, self.radius_km)
        dest_data = []
        for neighbor, dist in dest_neighbors:
            if neighbor in self.city_inbound_cpk:
                cpk_info = self.city_inbound_cpk[neighbor]
                cpk = cpk_info.get('inbound_cpk_mean', cpk_info) if isinstance(cpk_info, dict) else cpk_info
                dest_data.append({'cpk': cpk, 'dist': dist})
        
        predicted_cpk = None
        method = None
        confidence = 'Low'
        
        # Spatial IDW if enough neighbors
        if len(pickup_data) >= self.min_neighbors and len(dest_data) >= self.min_neighbors:
            epsilon = 1  # Avoid division by zero
            pickup_weights = [1 / (d['dist'] + epsilon) for d in pickup_data]
            pickup_weighted = sum(d['cpk'] * w for d, w in zip(pickup_data, pickup_weights)) / sum(pickup_weights)
            
            dest_weights = [1 / (d['dist'] + epsilon) for d in dest_data]
            dest_weighted = sum(d['cpk'] * w for d, w in zip(dest_data, dest_weights)) / sum(dest_weights)
            
            predicted_cpk = (pickup_weighted + dest_weighted) / 2
            method = f'Spatial IDW (R{self.radius_km})'
            confidence = 'Low'
        
        # Cascade V1: Region × Distance fallback
        if predicted_cpk is None and self.region_distance_cpk:
            p_region = CANONICAL_TO_REGION.get(p_can) or self.city_to_region.get(p_can)
            d_region = CANONICAL_TO_REGION.get(d_can) or self.city_to_region.get(d_can)
            
            # Get distance band
            band = None
            if distance_km and self.distance_quantiles:
                q = self.distance_quantiles
                if distance_km < q.get('q25', 0): band = 'q1'
                elif distance_km < q.get('q50', 0): band = 'q2'
                elif distance_km < q.get('q75', 0): band = 'q3'
                else: band = 'q4'
            
            if p_region and d_region and band:
                region_dist_cpk = self.region_distance_cpk.get((p_region, d_region, band))
                if region_dist_cpk is not None:
                    predicted_cpk = region_dist_cpk
                    method = f'Region×Distance ({band})'
                    confidence = 'Low'
        
        # Cascade V1: Distance Band fallback
        if predicted_cpk is None and self.distance_band_cpk:
            band = None
            if distance_km and self.distance_quantiles:
                q = self.distance_quantiles
                if distance_km < q.get('q25', 0): band = 'q1'
                elif distance_km < q.get('q50', 0): band = 'q2'
                elif distance_km < q.get('q75', 0): band = 'q3'
                else: band = 'q4'
            
            if band and band in self.distance_band_cpk:
                predicted_cpk = self.distance_band_cpk[band]
                method = f'Distance Band ({band})'
                confidence = 'Low'
        
        # Province P50 fallback
        if predicted_cpk is None:
            p_province = CANONICAL_TO_PROVINCE.get(p_can) or self.city_to_province.get(p_can)
            d_province = CANONICAL_TO_PROVINCE.get(d_can) or self.city_to_province.get(d_can)
            
            if p_province and d_province:
                prov_data = self.province_pair_cpk.get((p_province, d_province))
                if prov_data:
                    # Get the percentile value (e.g., 'p50')
                    percentile_key = f'p{self.percentile}'
                    if isinstance(prov_data, dict) and percentile_key in prov_data:
                        predicted_cpk = prov_data[percentile_key]
                    elif isinstance(prov_data, dict) and 'median' in prov_data:
                        predicted_cpk = prov_data['median']
                    elif isinstance(prov_data, (int, float)):
                        predicted_cpk = prov_data
                    
                    if predicted_cpk is not None:
                        method = f'Province P{self.percentile}'
                        confidence = 'Very Low'
        
        # Global mean fallback
        if predicted_cpk is None:
            predicted_cpk = self.global_mean
            method = 'Global Mean'
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
# REGIONAL MODEL (Legacy Fallback)
# Simple regional average model as ultimate fallback
# ============================================
class RegionalPredictor:
    """
    Simple regional CPK model as ultimate fallback.
    
    Uses 5-region matrix (Eastern, Western, Central, Northern, Southern)
    to provide a baseline estimate when other models fail.
    """
    
    def __init__(self, regional_cpk, city_to_region, global_mean=2.0):
        self.regional_cpk = regional_cpk
        self.city_to_region = city_to_region
        self.global_mean = global_mean
    
    def predict(self, pickup_city, dest_city, distance_km=None,
                pickup_region_override=None, dest_region_override=None):
        """Predict using regional averages."""
        p_can = CITY_TO_CANONICAL.get(pickup_city, pickup_city)
        d_can = CITY_TO_CANONICAL.get(dest_city, dest_city)
        
        # Get regions
        p_region = CANONICAL_TO_REGION.get(p_can) or self.city_to_region.get(p_can) or pickup_region_override
        d_region = CANONICAL_TO_REGION.get(d_can) or self.city_to_region.get(d_can) or dest_region_override
        
        predicted_cpk = None
        method = 'Regional'
        confidence = 'Very Low'
        
        if p_region and d_region:
            predicted_cpk = self.regional_cpk.get((p_region, d_region))
            if predicted_cpk is not None:
                method = 'Regional'
        
        if predicted_cpk is None:
            predicted_cpk = self.global_mean
            method = 'Global Mean'
        
        error_pct = ERROR_BARS.get(confidence, 0.35)
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
    
    # Load Index + Shrinkage model (90d Half-Life)
    # Try new filename first, then fall back to old filename for backward compatibility
    index_shrink_predictor = None
    index_shrink_paths = [
        os.path.join(MODEL_DIR, 'index_shrinkage_90hl.pkl'),  # New filename
        os.path.join(MODEL_DIR, 'rare_lane_models.pkl'),      # Legacy filename
    ]
    for index_path in index_shrink_paths:
        if os.path.exists(index_path):
            with open(index_path, 'rb') as f:
                index_artifacts = pickle.load(f)
            index_shrink_predictor = IndexShrinkagePredictor(index_artifacts)
            break
    
    # Load Spatial model (R150 P50 IDW) for new lanes
    # Try new filename first, then fall back to old blend model for backward compatibility
    spatial_predictor = None
    spatial_paths = [
        os.path.join(MODEL_DIR, 'spatial_r150_p50_idw.pkl'),  # New filename
        os.path.join(MODEL_DIR, 'new_lane_model_blend.pkl'),   # Legacy filename (will create SpatialPredictor with limited features)
    ]
    for spatial_path in spatial_paths:
        if os.path.exists(spatial_path):
            with open(spatial_path, 'rb') as f:
                spatial_artifacts = pickle.load(f)
            spatial_predictor = SpatialPredictor(spatial_artifacts)
            break
    
    # Load Port Pricing model (linear transforms for port loads)
    port_model = None
    port_model_path = os.path.join(MODEL_DIR, 'port_pricing_model.pkl')
    if os.path.exists(port_model_path):
        with open(port_model_path, 'rb') as f:
            port_model = pickle.load(f)
    
    # Build Regional predictor as ultimate fallback (from config data)
    regional_predictor = None
    city_to_region = config.get('city_to_region', {})
    if city_to_region:
        # Build regional CPK from historical data if available
        regional_cpk = {}
        # This will be populated at runtime from df_knn if available
        regional_predictor = RegionalPredictor(regional_cpk, city_to_region, global_mean=2.0)
    
    return {
        'carrier_model': carrier_model,
        'config': config,
        'df_knn': df_knn,
        'distance_matrix': distance_matrix,
        'index_shrink_predictor': index_shrink_predictor,
        'spatial_predictor': spatial_predictor,
        'regional_predictor': regional_predictor,
        'port_model': port_model
    }

models = load_models()
config = models['config']
df_knn = models['df_knn']
DISTANCE_MATRIX = models['distance_matrix']
index_shrink_predictor = models['index_shrink_predictor']
spatial_predictor = models['spatial_predictor']
regional_predictor = models['regional_predictor']
PORT_MODEL = models['port_model']


FEATURES = config['FEATURES']
ENTITY_MAPPING = config.get('ENTITY_MAPPING', 'Domestic')
DISTANCE_LOOKUP = config.get('DISTANCE_LOOKUP', {})

df_knn = df_knn[df_knn['entity_mapping'] == ENTITY_MAPPING].copy()

# ============================================
# LOAD PORT REFERENCE DATA (Actual Port Prices)
# ============================================
@st.cache_resource
def load_port_reference_data():
    """
    Load port reference data containing actual historical port prices.
    This data has real prices from port loads (Direct/Trip types).
    """
    # Check multiple possible locations
    possible_paths = [
        os.path.join(APP_DIR, 'port_reference_data.csv'),
        os.path.join(APP_DIR, 'model_export', 'port_reference_data.csv'),
    ]
    
    for port_csv_path in possible_paths:
        if os.path.exists(port_csv_path):
            df = pd.read_csv(port_csv_path)
            # Calculate days_ago for consistency with df_knn
            if 'pickup_date' in df.columns:
                df['pickup_date'] = pd.to_datetime(df['pickup_date'])
                df['days_ago'] = (pd.Timestamp.now() - df['pickup_date']).dt.days
            # Create lane column for lookups (multiple options for compatibility)
            if 'effective_lane' in df.columns:
                df['lane'] = df['effective_lane']
            elif 'pickup_city' in df.columns and 'effective_dest' in df.columns:
                df['lane'] = df['pickup_city'] + ' → ' + df['effective_dest']
            elif 'pickup_city' in df.columns and 'destination_city' in df.columns:
                df['lane'] = df['pickup_city'] + ' → ' + df['destination_city']
            return df
    return pd.DataFrame()

df_port = load_port_reference_data()

def get_port_lane_data(pickup_ar, dest_ar, load_type):
    """
    Get port lane data with proper matching logic.
    
    The updated port_reference_data.csv now has both domestic-style columns AND port-specific columns:
    - lane: pickup_city → destination_city (standard format)
    - effective_lane: pickup_city → effective_dest (port format)
    - effective_dest: The true drop-off location (important for Trip loads)
    
    For Port Direct: pickup_city → destination_city (normal domestic-style lane)
    For Port Roundtrip (Trip): pickup_city → effective_dest (true_drop_off is the real destination)
    
    Args:
        pickup_ar: Pickup city in Arabic
        dest_ar: Destination city in Arabic (user selected)
        load_type: 'Direct' or 'Trip'
    
    Returns:
        DataFrame of matching port loads
    """
    if len(df_port) == 0:
        return pd.DataFrame()
    
    # Filter by load_type first
    port_type_data = df_port[df_port['load_type'] == load_type] if 'load_type' in df_port.columns else df_port
    
    if len(port_type_data) == 0:
        return pd.DataFrame()
    
    port_lane_data = pd.DataFrame()
    
    # For DIRECT loads: Match by standard lane (pickup → destination)
    if load_type == 'Direct':
        # Strategy 1a: Match by lane column (pickup_city → destination_city)
        if 'lane' in port_type_data.columns:
            lane_ar = f"{pickup_ar} → {dest_ar}"
            port_lane_data = port_type_data[port_type_data['lane'] == lane_ar].copy()
        
        # Strategy 1b: Match by pickup_city and destination_city directly
        if len(port_lane_data) == 0 and 'destination_city' in port_type_data.columns:
            port_lane_data = port_type_data[
                (port_type_data['pickup_city'] == pickup_ar) &
                (port_type_data['destination_city'] == dest_ar)
            ].copy()
    
    # For TRIP loads: Match by effective_lane or effective_dest (the true drop-off)
    elif load_type == 'Trip':
        # Strategy 2a: Match by effective_lane (pickup_city → effective_dest)
        if 'effective_lane' in port_type_data.columns:
            effective_lane_ar = f"{pickup_ar} → {dest_ar}"
            port_lane_data = port_type_data[port_type_data['effective_lane'] == effective_lane_ar].copy()
        
        # Strategy 2b: Match by pickup_city and effective_dest (exact match)
        if len(port_lane_data) == 0 and 'effective_dest' in port_type_data.columns:
            port_lane_data = port_type_data[
                (port_type_data['pickup_city'] == pickup_ar) &
                (port_type_data['effective_dest'] == dest_ar)
            ].copy()
            
            # Try with English destination name
            if len(port_lane_data) == 0:
                dest_en = to_english_city(dest_ar)
                if dest_en:
                    port_lane_data = port_type_data[
                        (port_type_data['pickup_city'] == pickup_ar) &
                        (port_type_data['effective_dest'] == dest_en)
                    ].copy()
        
        # Strategy 2c: Normalized lookup on effective_dest (handles naming variations without fuzzy matching)
        # Normalize both the destination and effective_dest values for comparison
        if len(port_lane_data) == 0 and 'effective_dest' in port_type_data.columns:
            # Get the canonical for the destination city
            dest_canonical = CITY_TO_CANONICAL.get(dest_ar, dest_ar)
            dest_en = to_english_city(dest_ar)
            
            # Normalize destination for comparison
            dest_lower = normalize_english_text(dest_ar) if dest_ar else None
            dest_en_lower = normalize_english_text(dest_en) if dest_en else None
            dest_aggressive = normalize_aggressive(dest_ar)
            dest_en_aggressive = normalize_aggressive(dest_en) if dest_en else None
            dest_canonical_aggressive = normalize_aggressive(dest_canonical) if dest_canonical else None
            
            # Create normalized column for effective_dest and match
            def matches_destination(effective_dest_val):
                """Check if effective_dest matches the destination using normalized lookups."""
                if pd.isna(effective_dest_val):
                    return False
                ed_str = str(effective_dest_val).strip()
                
                # Exact match with canonical
                if ed_str == dest_canonical:
                    return True
                
                # Lookup effective_dest's canonical and compare
                ed_canonical = CITY_TO_CANONICAL.get(ed_str)
                if ed_canonical and ed_canonical == dest_canonical:
                    return True
                
                # Try lowercase normalized lookup
                ed_lower = normalize_english_text(ed_str)
                if ed_lower:
                    ed_canonical_from_lower = CITY_TO_CANONICAL_LOWER.get(ed_lower)
                    if ed_canonical_from_lower and ed_canonical_from_lower == dest_canonical:
                        return True
                    # Direct lowercase comparison
                    if ed_lower == dest_lower or ed_lower == dest_en_lower:
                        return True
                
                # Try aggressive normalized lookup (strips all punctuation)
                ed_aggressive = normalize_aggressive(ed_str)
                if ed_aggressive:
                    ed_canonical_from_agg = CITY_TO_CANONICAL_AGGRESSIVE.get(ed_aggressive)
                    if ed_canonical_from_agg and ed_canonical_from_agg == dest_canonical:
                        return True
                    # Direct aggressive comparison
                    if ed_aggressive == dest_aggressive or ed_aggressive == dest_en_aggressive or ed_aggressive == dest_canonical_aggressive:
                        return True
                
                return False
            
            # Apply matching function
            matching_mask = port_type_data['effective_dest'].apply(matches_destination)
            port_lane_data = port_type_data[
                (port_type_data['pickup_city'] == pickup_ar) & matching_mask
            ].copy()
    
    # Fallback: Try matching with any available lane/effective_lane columns regardless of load type
    if len(port_lane_data) == 0:
        lane_ar = f"{pickup_ar} → {dest_ar}"
        if 'effective_lane' in port_type_data.columns:
            port_lane_data = port_type_data[port_type_data['effective_lane'] == lane_ar].copy()
        if len(port_lane_data) == 0 and 'lane' in port_type_data.columns:
            port_lane_data = port_type_data[port_type_data['lane'] == lane_ar].copy()
    
    return port_lane_data


def get_port_ammunition_loads(pickup_ar, dest_ar, load_type, max_age_days=120):
    """
    Get ammunition loads from port data for display.
    
    Args:
        pickup_ar: Pickup city in Arabic
        dest_ar: Destination city in Arabic
        load_type: 'Direct' or 'Trip'
        max_age_days: Maximum age of loads to include
    
    Returns:
        DataFrame of recent port loads for this lane
    """
    port_lane_data = get_port_lane_data(pickup_ar, dest_ar, load_type)
    
    if len(port_lane_data) == 0:
        return pd.DataFrame()
    
    # Filter by age
    if 'days_ago' in port_lane_data.columns:
        port_lane_data = port_lane_data[port_lane_data['days_ago'] <= max_age_days]
    
    # Sort by recency
    if 'pickup_date' in port_lane_data.columns:
        port_lane_data = port_lane_data.sort_values('pickup_date', ascending=False)
    elif 'days_ago' in port_lane_data.columns:
        port_lane_data = port_lane_data.sort_values('days_ago', ascending=True)
    
    return port_lane_data.head(10)

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
# STARTUP VALIDATION: Check for unmapped port cities
# Cities in port_reference_data.csv effective_dest that don't have canonical mappings
# ============================================
def get_unmapped_port_cities():
    """
    Find cities in port reference data (effective_dest column) that have no canonical mapping.
    Uses normalized lookup (lowercase, aggressive normalization) but NOT fuzzy matching.
    
    Returns:
        List of unmapped city names from port reference data
    """
    if len(df_port) == 0 or 'effective_dest' not in df_port.columns:
        return []
    
    # Get unique effective_dest values from port data
    port_cities = set(df_port['effective_dest'].dropna().unique())
    
    unmapped = []
    for city in port_cities:
        city_str = str(city).strip()
        if not city_str or city_str.lower() in ('nan', 'none', ''):
            continue
        
        # Try exact match first
        if city_str in CITY_TO_CANONICAL:
            continue
        
        # Try lowercase normalized match (for English names)
        city_lower = normalize_english_text(city_str)
        if city_lower and city_lower in CITY_TO_CANONICAL_LOWER:
            continue
        
        # Try aggressive normalized match (strips all punctuation/spaces)
        city_aggressive = normalize_aggressive(city_str)
        if city_aggressive and city_aggressive in CITY_TO_CANONICAL_AGGRESSIVE:
            continue
        
        # No match found - this is an unmapped city
        unmapped.append(city_str)
    
    return sorted(set(unmapped))

PORT_CITIES_UNMAPPED = get_unmapped_port_cities()

# ============================================
# DROPDOWN SETUP
# ============================================
all_canonicals = sorted(list(set(CITY_TO_CANONICAL.values())))
pickup_cities_en = sorted(list(set([to_english_city(c) for c in all_canonicals])))
dest_cities_en = pickup_cities_en
vehicle_types_en = sorted(set(VEHICLE_TYPE_EN.values()))
# Safe commodity extraction handling mixed types/NaNs
unique_comms = [c for c in df_knn['commodity'].unique() if pd.notna(c)]
commodities = sorted(set([to_english_commodity(str(c)) for c in unique_comms if str(c).strip() != '']))

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

def apply_port_transform(domestic_price, truck_type):
    """
    Apply port pricing linear transform.
    Port = alpha × Domestic + beta
    
    Args:
        domestic_price: Base domestic price (SAR)
        truck_type: 'Domestic', 'Port Direct', or 'Port Roundtrip'
    
    Returns:
        Transformed price (same as domestic if Domestic type or no port model)
    """
    if domestic_price is None or domestic_price <= 0:
        return domestic_price
    
    if truck_type == 'Domestic' or not PORT_MODEL:
        return domestic_price
    
    transforms = PORT_MODEL.get('recommended', {})
    
    if truck_type == 'Port Direct':
        params = transforms.get('Direct', transforms.get('fallback', {}))
    elif truck_type == 'Port Roundtrip':
        params = transforms.get('Trip', transforms.get('fallback', {}))
    else:
        return domestic_price
    
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 0)
    
    return round(alpha * domestic_price + beta)

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
    # SKIP if either city has user-provided coordinates (Google Maps won't find it)
    if pickup_en and dest_en:
        should_log = True
        
        # Skip Google Maps for cities added manually with coordinates
        # Check city_resolutions directly (only during bulk wizard)
        try:
            if 'city_resolutions' in st.session_state:
                resolutions = st.session_state.city_resolutions
                # Check if pickup or dest is a new_city (user-provided coordinates)
                pickup_res = resolutions.get(pickup_ar) or resolutions.get(p_can)
                dest_res = resolutions.get(dest_ar) or resolutions.get(d_can)
                if (pickup_res and pickup_res.get('type') == 'new_city') or \
                   (dest_res and dest_res.get('type') == 'new_city'):
                    should_log = False  # User must input distance manually in Step 2
        except:
            pass  # Not in bulk wizard context
        
        # If check_history is True, only log if at least one city has historical data
        # (Used by Master Grid to avoid logging routes where neither city has any history)
        if check_history and should_log:
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
    1. RECENCY (90d): If recent loads exist (within RECENCY_WINDOW days), use median
    2. INDEX+SHRINKAGE (90d HL): If ANY historical data exists for this lane, use Index+Shrinkage model
    3. SPATIAL R150 P50 IDW: For NEW lanes (no history), use spatial neighbor interpolation
    4. REGIONAL: Fallback to 5-region averages if spatial fails
    5. DEFAULT: Last resort - use default CPK × distance
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
        # 1. RECENCY MODEL (90d): Use median of recent loads
        base_price = flatbed_data[flatbed_data['days_ago'] <= RECENCY_WINDOW]['total_carrier_price'].median()
        model = 'Recency (90d)'
        conf = 'High' if recent_count >= 5 else ('Medium' if recent_count >= 2 else 'Low')
        
    elif index_shrink_predictor and index_shrink_predictor.has_lane_data(lane):
        # 2. INDEX + SHRINKAGE (90d HL): Lane has ANY historical data (even if not recent)
        # Uses exponential decay weighting with 90-day half-life
        p = index_shrink_predictor.predict(pickup_ar, dest_ar, distance_km)
        base_price, model, conf = p.get('predicted_cost'), f"Index+Shrink (90d HL)", p['confidence']
        
    elif spatial_predictor:
        # 3. SPATIAL R150 P50 IDW: NEW lane with NO historical data
        # Uses inverse-distance weighted CPK from nearby cities within 150km
        # Falls back to Province P50 if not enough neighbors
        p = spatial_predictor.predict(pickup_ar, dest_ar, distance_km,
                                      pickup_region_override=pickup_region_override,
                                      dest_region_override=dest_region_override)
        base_price, model, conf = p.get('predicted_cost'), p['method'], p['confidence']
        
    elif regional_predictor:
        # 4. REGIONAL FALLBACK: Use 5-region averages
        p = regional_predictor.predict(pickup_ar, dest_ar, distance_km,
                                       pickup_region_override=pickup_region_override,
                                       dest_region_override=dest_region_override)
        base_price, model, conf = p.get('predicted_cost'), p['method'], p['confidence']
        
    else:
        # 5. DEFAULT: Last resort fallback (no models available)
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

def price_single_route(pickup_ar, dest_ar, vehicle_ar=None, commodity=None, weight=None, truck_type='Domestic'):
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
    
    # Apply port pricing transform if not Domestic
    buy_price = pricing['buy_price']
    sell_price = pricing['sell_price']
    model_used = pricing['model_used']
    
    # Prepare historical/recent values (may be transformed for port)
    disp_h_min, disp_h_med, disp_h_max = h_min, h_med, h_max
    disp_r_min, disp_r_med, disp_r_max = r_min, r_med, r_max
    disp_hs_min, disp_hs_med, disp_hs_max = hs_min, hs_med, hs_max
    disp_rs_min, disp_rs_med, disp_rs_max = rs_min, rs_med, rs_max
    
    if truck_type != 'Domestic' and PORT_MODEL:
        # ═══════════════════════════════════════════════════════════════════════════
        # PORT PRICING CASCADE:
        # 1. Recency (90d): If port data exists for this lane in last 90 days → use median
        # 2. Linear Transform: Otherwise → apply linear transform to domestic prediction
        # ═══════════════════════════════════════════════════════════════════════════
        
        load_type_filter = 'Direct' if truck_type == 'Port Direct' else 'Trip'
        
        # Get port lane data using the helper function
        port_lane_data = get_port_lane_data(pickup_ar, dest_ar, load_type_filter)
        
        # Check for recent port data (Recency model)
        port_rec_data = pd.DataFrame()
        if len(port_lane_data) > 0 and 'days_ago' in port_lane_data.columns:
            port_rec_data = port_lane_data[port_lane_data['days_ago'] <= RECENCY_WINDOW]
        elif len(port_lane_data) > 0:
            port_rec_data = port_lane_data  # No days_ago column, use all data
        
        # CASCADE LOGIC
        if len(port_rec_data) > 0:
            # RECENCY MODEL: Use median of recent port data
            buy_price = round_to_nearest(port_rec_data['total_carrier_price'].median(), 100)
            model_used = f"Port Recency ({len(port_rec_data)} loads)"
        else:
            # LINEAR TRANSFORM: Apply port transform to domestic prediction
            buy_price = apply_port_transform(buy_price, truck_type)
            buy_price = round_to_nearest(buy_price, 100)
            model_used = f"Domestic {model_used} + Port Transform"
        
        # Apply margin after price determination
        if buy_price:
            _, margin, _ = get_backhaul_probability(dest_ar)
            sell_price = round_to_nearest(buy_price * (1 + margin), 50)
        
        # Update display stats with actual port data
        if len(port_lane_data) > 0:
            disp_h_min = round(port_lane_data['total_carrier_price'].min(), 0)
            disp_h_med = round(port_lane_data['total_carrier_price'].median(), 0)
            disp_h_max = round(port_lane_data['total_carrier_price'].max(), 0)
            h_count = len(port_lane_data)
            
            if len(port_rec_data) > 0:
                disp_r_min = round(port_rec_data['total_carrier_price'].min(), 0)
                disp_r_med = round(port_rec_data['total_carrier_price'].median(), 0)
                disp_r_max = round(port_rec_data['total_carrier_price'].max(), 0)
                r_count = len(port_rec_data)
            else:
                disp_r_min, disp_r_med, disp_r_max, r_count = None, None, None, 0
            
            # Check if port data has shipper prices (now included in updated port_reference_data.csv)
            if 'total_shipper_price' in port_lane_data.columns and port_lane_data['total_shipper_price'].notna().any():
                disp_hs_min = round(port_lane_data['total_shipper_price'].min(), 0)
                disp_hs_med = round(port_lane_data['total_shipper_price'].median(), 0)
                disp_hs_max = round(port_lane_data['total_shipper_price'].max(), 0)
                if len(port_rec_data) > 0 and 'total_shipper_price' in port_rec_data.columns:
                    disp_rs_min = round(port_rec_data['total_shipper_price'].min(), 0)
                    disp_rs_med = round(port_rec_data['total_shipper_price'].median(), 0)
                    disp_rs_max = round(port_rec_data['total_shipper_price'].max(), 0)
                else:
                    disp_rs_min, disp_rs_med, disp_rs_max = None, None, None
            else:
                # Port data doesn't have shipper prices - set to None
                disp_hs_min, disp_hs_med, disp_hs_max = None, None, None
                disp_rs_min, disp_rs_med, disp_rs_max = None, None, None
            
            # Override reference sell with port data
            if 'total_shipper_price' in port_lane_data.columns and port_lane_data['total_shipper_price'].notna().any():
                port_recent = port_lane_data[port_lane_data['days_ago'] <= RECENCY_WINDOW] if 'days_ago' in port_lane_data.columns else port_lane_data
                if len(port_recent) > 0 and 'total_shipper_price' in port_recent.columns and port_recent['total_shipper_price'].notna().any():
                    pricing['ref_sell_price'] = round(port_recent['total_shipper_price'].median(), 0)
                    pricing['ref_sell_source'] = f'Port Recent {RECENCY_WINDOW}d'
                elif port_lane_data['total_shipper_price'].notna().any():
                    pricing['ref_sell_price'] = round(port_lane_data['total_shipper_price'].median(), 0)
                    pricing['ref_sell_source'] = 'Port Historical'
        else:
            # No port data found - keep domestic display values but update counts
            # This allows showing domestic reference data when using port transform
            h_count = 0  # Indicate no port-specific data
            r_count = 0
            # Keep disp_h_*, disp_r_*, disp_hs_*, disp_rs_* as domestic values for reference
            # But mark them as domestic-derived by not changing them
            # Override reference sell source to indicate it's domestic-derived
            if pricing.get('ref_sell_source'):
                pricing['ref_sell_source'] = f"Domestic {pricing['ref_sell_source']}"
    
    res = {
        'Freight_Segment': truck_type,
        'Vehicle_Type': to_english_vehicle(vehicle_ar), 'Commodity': to_english_commodity(commodity),
        'Weight_Tons': round(weight, 1), 'Distance_km': round(dist, 0), 'Distance_Source': dist_src,
        'Hist_Count': h_count, 'Hist_Min': disp_h_min, 'Hist_Median': disp_h_med, 'Hist_Max': disp_h_max,
        f'Recent_{RECENCY_WINDOW}d_Count': r_count, f'Recent_{RECENCY_WINDOW}d_Min': disp_r_min, f'Recent_{RECENCY_WINDOW}d_Median': disp_r_med, f'Recent_{RECENCY_WINDOW}d_Max': disp_r_max,
        'Hist_Sell_Min': disp_hs_min, 'Hist_Sell_Median': disp_hs_med, 'Hist_Sell_Max': disp_hs_max,
        f'Recent_{RECENCY_WINDOW}d_Sell_Min': disp_rs_min, f'Recent_{RECENCY_WINDOW}d_Sell_Median': disp_rs_med, f'Recent_{RECENCY_WINDOW}d_Sell_Max': disp_rs_max,
        'Buy_Price': buy_price, 'Rec_Sell_Price': sell_price,
        'Ref_Sell_Price': pricing['ref_sell_price'], 'Ref_Sell_Source': pricing['ref_sell_source'],
        'Rental_Cost': pricing['rental_cost'], 'Target_Margin': pricing['target_margin'],
        'Backhaul_Probability': pricing['backhaul_probability'], 'Backhaul_Ratio': pricing['backhaul_ratio'],
        'Model_Used': model_used, 'Confidence': pricing['confidence'],
        'Cost_Per_KM': round(buy_price/dist, 2) if buy_price and dist > 0 else None,
        'Is_Rare_Lane': r_count == 0
    }
    
    if index_shrink_predictor:
        p = index_shrink_predictor.predict(pickup_ar, dest_ar, dist)
        res.update({'IndexShrink_Price': p.get('predicted_cost'), 'IndexShrink_Upper': p.get('cost_high'), 'IndexShrink_Method': p['method']})
    if spatial_predictor:
        p = spatial_predictor.predict(pickup_ar, dest_ar, dist)
        res.update({'Spatial_Price': p.get('predicted_cost'), 'Spatial_Upper': p.get('cost_high'), 'Spatial_Method': p['method']})
    return res

def lookup_route_stats(pickup_ar, dest_ar, vehicle_ar=None, dist_override=None, check_history=False,
                       pickup_region_override=None, dest_region_override=None, weight=None, truck_type='Domestic'):
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
        truck_type: 'Domestic', 'Port Direct', or 'Port Roundtrip' (default: 'Domestic')
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
    
    # Apply port pricing transform if not Domestic
    buy_price = pricing['buy_price']
    sell_price = pricing['sell_price']
    model_used = pricing['model_used']
    
    if truck_type != 'Domestic' and PORT_MODEL:
        # ═══════════════════════════════════════════════════════════════════════════
        # PORT PRICING CASCADE (same as price_single_route):
        # 1. Recency (90d): If port data exists → use median
        # 2. Linear Transform: Otherwise → apply linear transform to domestic prediction
        # ═══════════════════════════════════════════════════════════════════════════
        
        load_type_filter = 'Direct' if truck_type == 'Port Direct' else 'Trip'
        
        # Get port lane data
        port_lane_data = get_port_lane_data(pickup_ar, dest_ar, load_type_filter)
        
        # Check for recent port data
        port_rec_data = pd.DataFrame()
        if len(port_lane_data) > 0 and 'days_ago' in port_lane_data.columns:
            port_rec_data = port_lane_data[port_lane_data['days_ago'] <= RECENCY_WINDOW]
        elif len(port_lane_data) > 0:
            port_rec_data = port_lane_data
        
        # CASCADE LOGIC
        if len(port_rec_data) > 0:
            # RECENCY MODEL: Use median of recent port data
            buy_price = round_to_nearest(port_rec_data['total_carrier_price'].median(), 100)
            model_used = f"Port Recency ({len(port_rec_data)})"
        else:
            # LINEAR TRANSFORM: Apply port transform to domestic prediction
            buy_price = apply_port_transform(buy_price, truck_type)
            buy_price = round_to_nearest(buy_price, 100)
            model_used = f"{model_used} + Port Transform"
        
        # Apply margin after price determination
        if buy_price:
            _, margin, _ = get_backhaul_probability(dest_ar)
            sell_price = round_to_nearest(buy_price * (1 + margin), 50)
    
    return {
        'From': to_english_city(pickup_ar), 
        'To': to_english_city(dest_ar), 
        'Freight_Segment': truck_type,
        'Vehicle': to_english_vehicle(vehicle_ar), 
        'Weight_Tons': weight if weight else 25.0,
        'Distance': int(dist) if dist else 0,
        'Distance_Source': dist_source,
        'Buy_Price': buy_price, 'Rec_Sell': sell_price,
        'Ref_Sell': pricing['ref_sell_price'], 'Ref_Sell_Src': pricing['ref_sell_source'],
        'Rental_Cost': pricing['rental_cost'], 'Margin': pricing['target_margin'],
        'Model': model_used, 'Confidence': pricing['confidence'], 'Recent_N': pricing['recent_count']
    }
                           
# ============================================
# APP UI
# ============================================
st.title("🚚 Freight Pricing Tool")
model_status = []
if index_shrink_predictor: model_status.append("✅ Index+Shrink (90d HL)")
if spatial_predictor: model_status.append("✅ Spatial R150 IDW")
if regional_predictor: model_status.append("✅ Regional Fallback")
if PORT_MODEL: model_status.append("✅ Port Transform")
if len(df_port) > 0: 
    port_direct = len(df_port[df_port['load_type'] == 'Direct']) if 'load_type' in df_port.columns else 0
    port_trip = len(df_port[df_port['load_type'] == 'Trip']) if 'load_type' in df_port.columns else 0
    model_status.append(f"🚢 Port Data ({port_direct}D/{port_trip}T)")
dist_status = f"✅ {len(DISTANCE_MATRIX):,} distances" if DISTANCE_MATRIX else ""
st.caption(f"ML-powered pricing | Domestic + Port | {' | '.join(model_status)} | {dist_status}")

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

# Warning for cities in port reference data but not in normalization CSV
if PORT_CITIES_UNMAPPED:
    with st.expander(f"🚢 {len(PORT_CITIES_UNMAPPED)} port cities in effective_dest missing from canonicals CSV", expanded=False):
        st.warning("""
        **Action Required:** These cities appear in the port_reference_data.csv `effective_dest` column but have NO mapping in city_normalization.csv.
        Port pricing lookups may fail for lanes involving these cities.
        """)
        for city in PORT_CITIES_UNMAPPED:
            st.code(city)
        st.info("Add these as variants to city_normalization.csv with columns: variant, canonical, english, region")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Route Pricing", "📦 Bulk Route Lookup", "🗺️ Map Explorer", "🔧 Admin"])

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
    if 'single_truck_type' not in st.session_state:
        st.session_state.single_truck_type = DEFAULT_TRUCK_TYPE
    
    # Use columns with swap button in the middle
    col1, col_swap, col2, col3, col4 = st.columns([3, 0.5, 3, 2.5, 2])
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
    with col4:
        truck_type = st.selectbox("Freight Segment", options=TRUCK_TYPES, key='single_truck_type', 
                                   help="Domestic: Standard pricing | Port Direct/Indirect: Port pricing transform")

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
        result = price_single_route(pickup_city, destination_city, vehicle_type, comm_in, weight, truck_type)
        st.session_state.last_result, st.session_state.last_lane = result, {'pickup_ar': pickup_city, 'dest_ar': destination_city, 'pickup_en': pickup_en, 'dest_en': dest_en}
        
        if result['Distance_km'] == 0 or result['Distance_Source'] == 'Default': st.error(f"⚠️ Distance data missing or estimated ({result['Distance_km']} km)")
        
        st.header("💰 Pricing Results")
        truck_type_display = result.get('Freight_Segment', 'Domestic')
        truck_icon = "🏠" if truck_type_display == 'Domestic' else "🚢"
        st.info(f"**{pickup_en} → {dest_en}** | {truck_icon} {truck_type_display} | 🚛 {result['Vehicle_Type']} | 📏 {result['Distance_km']:.0f} km | ⚖️ {result['Weight_Tons']:.1f} T")
        
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
            truck_type_for_comparison = result.get('Freight_Segment', 'Domestic')
            
            if 'IndexShrink_Price' in result and result['IndexShrink_Price']:
                idx_price = result['IndexShrink_Price']
                idx_upper = result['IndexShrink_Upper']
                # Apply port transform if not Domestic
                if truck_type_for_comparison != 'Domestic' and PORT_MODEL:
                    idx_price = apply_port_transform(idx_price, truck_type_for_comparison)
                    idx_upper = apply_port_transform(idx_upper, truck_type_for_comparison) if idx_upper else None
                model_df.append({
                    'Model': f"{result['IndexShrink_Method']}" + (f" + Port" if truck_type_for_comparison != 'Domestic' else ""), 
                    'Prediction': f"{idx_price:,.0f} SAR", 
                    'Upper Bound': f"{idx_upper:,.0f} SAR" if idx_upper else "—"
                })
            if 'Blend_Price' in result and result['Blend_Price']:
                blend_price = result['Blend_Price']
                blend_upper = result['Blend_Upper']
                # Apply port transform if not Domestic
                if truck_type_for_comparison != 'Domestic' and PORT_MODEL:
                    blend_price = apply_port_transform(blend_price, truck_type_for_comparison)
                    blend_upper = apply_port_transform(blend_upper, truck_type_for_comparison) if blend_upper else None
                model_df.append({
                    'Model': f"{result['Blend_Method']}" + (f" + Port" if truck_type_for_comparison != 'Domestic' else ""), 
                    'Prediction': f"{blend_price:,.0f} SAR", 
                    'Upper Bound': f"{blend_upper:,.0f} SAR" if blend_upper else "—"
                })
            if 'Spatial_Price' in result and result['Spatial_Price']:
                spatial_price = result['Spatial_Price']
                spatial_upper = result.get('Spatial_Upper')
                # Apply port transform if not Domestic
                if truck_type_for_comparison != 'Domestic' and PORT_MODEL:
                    spatial_price = apply_port_transform(spatial_price, truck_type_for_comparison)
                    spatial_upper = apply_port_transform(spatial_upper, truck_type_for_comparison) if spatial_upper else None
                model_df.append({
                    'Model': f"{result['Spatial_Method']}" + (f" + Port" if truck_type_for_comparison != 'Domestic' else ""), 
                    'Prediction': f"{spatial_price:,.0f} SAR", 
                    'Upper Bound': f"{spatial_upper:,.0f} SAR" if spatial_upper else "—"
                })
            if model_df: st.dataframe(pd.DataFrame(model_df), use_container_width=True, hide_index=True)
            
        if not result['Is_Rare_Lane']:
            st.markdown("---")
            st.subheader("🚚 Your Ammunition (Recent Matches)")
            
            truck_type_result = result.get('Freight_Segment', 'Domestic')
            
            if truck_type_result != 'Domestic' and len(df_port) > 0:
                # ═══════════════════════════════════════════════════════════════════
                # PORT DATA: Show actual port loads
                # ═══════════════════════════════════════════════════════════════════
                load_type = 'Direct' if truck_type_result == 'Port Direct' else 'Trip'
                port_samples = get_port_ammunition_loads(pickup_city, destination_city, load_type)
                
                if len(port_samples) > 0:
                    st.markdown(f"**Port {load_type} Loads:**")
                    
                    # Build display dataframe
                    display_cols = ['pickup_date', 'total_carrier_price', 'days_ago']
                    col_names = ['Date', 'Carrier (SAR)', 'Days Ago']
                    
                    # Add effective_dest if available (shows true destination)
                    if 'effective_dest' in port_samples.columns:
                        port_samples['Destination'] = port_samples['effective_dest']
                        display_cols.insert(1, 'Destination')
                        col_names.insert(1, 'Destination')
                    
                    # Add distance if available
                    if 'distance' in port_samples.columns:
                        display_cols.insert(-1, 'distance')
                        col_names.insert(-1, 'Distance (km)')
                    
                    display_port = port_samples[display_cols].copy()
                    display_port.columns = col_names
                    
                    if 'Date' in display_port.columns:
                        display_port['Date'] = pd.to_datetime(display_port['Date']).dt.strftime('%Y-%m-%d')
                    display_port['Carrier (SAR)'] = display_port['Carrier (SAR)'].round(0).astype(int)
                    if 'Distance (km)' in display_port.columns:
                        display_port['Distance (km)'] = display_port['Distance (km)'].round(0).astype(int)
                    
                    st.dataframe(display_port, use_container_width=True, hide_index=True)
                    st.caption(f"**{len(port_samples)} port loads** | Data from port_reference_data.csv")
                else:
                    st.caption(f"No recent port {load_type} loads found for this lane")
                    st.info("💡 Using linear transform of domestic prices as fallback")
            else:
                # ═══════════════════════════════════════════════════════════════════
                # DOMESTIC DATA: Show domestic loads (original behavior)
                # ═══════════════════════════════════════════════════════════════════
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
    # Centered Header
    st.markdown("<h2 style='text-align: center;'>📦 Bulk Route Lookup</h2>", unsafe_allow_html=True)
    
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
        
        # Explicitly clear city resolutions
        if 'city_resolutions' in st.session_state:
            del st.session_state['city_resolutions']
    
    # Show current step indicator
    # UPDATED: Renamed steps and centered layout
    steps = ["Step 1: Upload", "Step 2: Cities", "Step 3: Distances", "Step 4: Pricing"]
    current_step = st.session_state.bulk_wizard_step
    
    # Progress indicator - Centered using spacer columns
    _, c_prog, _ = st.columns([1, 10, 1])
    
    with c_prog:
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
        st.markdown("<h3 style='text-align: center;'>Step 1: Upload Data</h3>", unsafe_allow_html=True)
        
        # 1. Description and Template Button
        st.markdown("""
        <div style='text-align: center; margin-bottom: 10px;'>
        Prepare your CSV with the following columns (order matters, names ignored):<br>
        <b>1. Pickup City</b> &nbsp;|&nbsp; <b>2. Destination City</b> &nbsp;|&nbsp; <b>3. Vehicle</b> (Optional) &nbsp;|&nbsp; <b>4. Distance</b> (Optional)
        </div>
        """, unsafe_allow_html=True)
        
        # Centered Template Button
        c_fill1, c_temp, c_fill2 = st.columns([1, 2, 1])
        with c_temp:
            if st.button("📋 Open Google Sheets Template", use_container_width=True):
                template_df = pd.DataFrame({
                    'Pickup': ['Jeddah', 'Riyadh', 'Unknown City'], 
                    'Destination': ['Riyadh', 'Dammam', 'Jeddah'], 
                    'Vehicle_Type': ['Flatbed Trailer', '', ''],
                    'Distance': ['', '', '450']
                })
                success, msg = populate_rfq_template_sheet(template_df)
                if success:
                    rfq_url = st.secrets.get('RFQ_url', '')
                    st.toast(msg, icon="✅")
                    if rfq_url:
                        st.markdown(f"<div style='text-align: center'><a href='{rfq_url}' target='_blank'>🔗 Click here to open Sheet</a></div>", unsafe_allow_html=True)
                else:
                    st.error(msg)
        
        st.markdown("---")

        # 2. File Uploader
        upl = st.file_uploader("Upload CSV File", type=['csv'], key='bulk_csv_upload', label_visibility="collapsed")
        
        if not upl:
            st.info("👆 Please upload a CSV file to begin.")

        if upl:
            try:
                r_df = pd.read_csv(upl)
                
                # Layout: Data Preview on left, User Details on right
                col_data, col_details = st.columns([2, 1])
                
                with col_data:
                    st.caption(f"✅ Loaded {len(r_df)} rows")
                    st.dataframe(r_df.head(5), use_container_width=True, height=150)

                with col_details:
                    st.markdown("##### 👤 User Details")
                    username = st.text_input("Enter Your Name", key='bulk_username', 
                                              placeholder="Required for tracking",
                                              help="Your name will be logged with any changes you make")
                    if not username:
                        st.caption("⚠️ Name required to proceed")
                
                st.markdown("---")

                # 3. Vehicle Selection
                st.markdown("<h5 style='text-align: center;'>🚛 Select Vehicles to Price</h5>", unsafe_allow_html=True)
                
                # Centered checkboxes
                v_cols = st.columns(4)
                selected_vehicles = []
                for i, v_en in enumerate(vehicle_types_en):
                    with v_cols[i % 4]:
                        if st.checkbox(v_en, value=False, key=f'vtype_{v_en}'):
                            selected_vehicles.append(v_en)
                
                # Only warn if CSV has no vehicle column
                has_vehicle_col = len(r_df.columns) > 2
                if not selected_vehicles and not has_vehicle_col:
                    st.warning("⚠️ Please select at least one vehicle type OR include a vehicle column in your CSV.")
                elif not selected_vehicles:
                    st.info("💡 No vehicles selected - will use vehicle from your CSV (Column 3).")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # 3b. Freight Segment Selection (Port Pricing)
                st.markdown("<h5 style='text-align: center;'>Select Freight Segment</h5>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; font-size: 0.85rem; color: gray;'>Freight segment applies port pricing transform to all routes in this batch</p>", unsafe_allow_html=True)
                
                # Use columns like the vehicle checkboxes above
                seg_cols = st.columns([1, 1, 1, 1, 1])  # 5 columns: spacer + 3 options + spacer
                
                # Initialize the selected truck type in session state if not present
                if 'bulk_truck_type' not in st.session_state:
                    st.session_state.bulk_truck_type = 'Domestic'
                
                with seg_cols[1]:
                    if st.button("🏠 Domestic", use_container_width=True, 
                                 type="primary" if st.session_state.bulk_truck_type == 'Domestic' else "secondary"):
                        st.session_state.bulk_truck_type = 'Domestic'
                        st.rerun()
                with seg_cols[2]:
                    if st.button("🚢 Port Direct", use_container_width=True,
                                 type="primary" if st.session_state.bulk_truck_type == 'Port Direct' else "secondary"):
                        st.session_state.bulk_truck_type = 'Port Direct'
                        st.rerun()
                with seg_cols[3]:
                    if st.button("📦 Port Roundtrip", use_container_width=True,
                                 type="primary" if st.session_state.bulk_truck_type == 'Port Roundtrip' else "secondary"):
                        st.session_state.bulk_truck_type = 'Port Roundtrip'
                        st.rerun()
                
                selected_truck_type = st.session_state.bulk_truck_type


                st.markdown("<br>", unsafe_allow_html=True)

                # 4. Action Buttons (Equal Width for balance)
                btn_col1, btn_col2 = st.columns(2)
                
                with btn_col1:
                    if st.button("🔄 Reset / Clear", use_container_width=True):
                        reset_wizard()
                        st.rerun()

                with btn_col2:
                    # Logic to disable button if requirements aren't met
                    # Relax requirement: Allow empty selected_vehicles IF CSV has vehicle column (column 3)
                    has_vehicle_col = len(r_df.columns) > 2
                    is_ready = username and (selected_vehicles or has_vehicle_col)
                    
                    if st.button("▶️ Analyze Cities", type="primary", use_container_width=True, disabled=not is_ready):
                        if not username:
                            st.error("Please enter your name above.")
                        elif not selected_vehicles and not has_vehicle_col:
                            st.error("Please select at least one vehicle type OR ensure your CSV has a vehicle column (Col 3).")
                        else:
                            # --- PROCESSING LOGIC STARTS HERE ---
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("Step 1/4: Parsing uploaded file...")
                            unmatched_cities = {} 
                            parsed_rows = []
                            
                            for i in range(len(r_df)):
                                row = r_df.iloc[i]
                                p_raw = str(row.iloc[0]).strip() if len(row) > 0 else ''
                                d_raw = str(row.iloc[1]).strip() if len(row) > 1 else ''
                                
                                # Parse vehicle (column 3 - NEW ORDER)
                                v_raw = ''
                                if len(row) > 2:
                                    v_raw = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ''
                                
                                # Parse distance (column 4 - NEW ORDER)
                                dist_override = None
                                if len(row) > 3:
                                    try:
                                        val = row.iloc[3]
                                        if pd.notna(val) and str(val).strip() not in ['', 'nan', 'None']:
                                            dist_override = float(str(val).replace(',', '').strip())
                                    except (ValueError, TypeError):
                                        pass
                                
                                # Normalize cities
                                p_ar, p_ok = normalize_city(p_raw)
                                d_ar, d_ok = normalize_city(d_raw)
                                
                                # Track unmatched
                                if not p_ok and p_raw:
                                    if p_raw not in unmatched_cities:
                                        unmatched_cities[p_raw] = {'rows': [], 'col': 'Pickup'}
                                    unmatched_cities[p_raw]['rows'].append(i + 1)
                                    
                                    log_exception('unmatched_city', {
                                        'pickup_city': p_raw, 'destination_city': '', 'original_value': p_raw, 
                                        'column': 'Pickup', 'row_num': i + 1
                                    }, immediate=False)
                                
                                if not d_ok and d_raw:
                                    if d_raw not in unmatched_cities:
                                        unmatched_cities[d_raw] = {'rows': [], 'col': 'Destination'}
                                    unmatched_cities[d_raw]['rows'].append(i + 1)
                                
                                    log_exception('unmatched_city', {
                                        'pickup_city': '', 'destination_city': d_raw, 'original_value': d_raw, 
                                        'column': 'Destination', 'row_num': i + 1
                                    }, immediate=False)
                                
                                parsed_rows.append({
                                    'row_num': i + 1, 'pickup_raw': p_raw, 'dest_raw': d_raw,
                                    'pickup_ar': p_ar, 'dest_ar': d_ar, 'pickup_ok': p_ok, 'dest_ok': d_ok,
                                    'dist_override': dist_override, 'vehicle_raw': v_raw
                                })
                            
                            progress_bar.progress(30)
                            status_text.text("Step 2/4: Validating matched cities...")
                            
                            # Log missing distances for matched cities immediately
                            matched_pairs_checked = set()
                            for row in parsed_rows:
                                if row['pickup_ok'] and row['dest_ok'] and not row.get('dist_override'):
                                    pair_key = (row['pickup_ar'], row['dest_ar'])
                                    if pair_key not in matched_pairs_checked:
                                        matched_pairs_checked.add(pair_key)
                                        get_distance(row['pickup_ar'], row['dest_ar'], immediate_log=False)
                            
                            progress_bar.progress(50)
                            status_text.text("Step 3/4: Saving logs to Google Sheets (this may take a few seconds)...")
                            
                            # Flush logs
                            dist_flushed_ok, dist_flushed_count = flush_matched_distances_to_sheet()
                            err_flushed_ok, err_flushed_count = flush_error_log_to_sheet()
                            
                            msgs = []
                            if dist_flushed_count > 0: msgs.append(f"✅ Logged {dist_flushed_count} missing distances")
                            if err_flushed_count > 0: msgs.append(f"⚠️ Logged {err_flushed_count} errors")
                            st.session_state['last_distance_flush'] = " | ".join(msgs) if msgs else "ℹ️ No new issues to log"
                            
                            # Store in session state
                            st.session_state.bulk_wizard_data = {
                                'username': username, 'selected_vehicles': selected_vehicles,
                                'selected_truck_type': selected_truck_type,
                                'parsed_rows': parsed_rows, 'unmatched_cities': unmatched_cities,
                                'original_df': r_df
                            }
                            
                            # Move to next step
                            if unmatched_cities:
                                # Run fuzzy matching
                                progress_bar.progress(70)
                                status_text.text("Step 4/4: Analyzing unmatched cities...")
                                fuzzy_results = {}
                                if RAPIDFUZZ_AVAILABLE:
                                    fuzzy_results = batch_fuzzy_match_cities(list(unmatched_cities.keys()), threshold=80)
                                    st.session_state.bulk_wizard_data['fuzzy_results'] = fuzzy_results
                                
                                # Google Geocoding for poor matches
                                cities_needing_geocode = []
                                for city_name in unmatched_cities.keys():
                                    fuzzy = fuzzy_results.get(city_name, {})
                                    if not fuzzy.get('match_found') or fuzzy.get('confidence', 0) < 80:
                                        cities_needing_geocode.append(city_name)
                                
                                if cities_needing_geocode:
                                    with st.spinner(f"🌍 Geocoding {len(cities_needing_geocode)} cities via Google Maps..."):
                                        success, count_written, start_row = write_cities_for_geocoding(cities_needing_geocode)
                                        if success:
                                            import time
                                            time.sleep(3) # Wait for sheets formula
                                            google_results = read_geocode_results(cities_needing_geocode)
                                            st.session_state.bulk_wizard_data['google_suggestions'] = google_results
                                        else:
                                            st.session_state.bulk_wizard_data['google_suggestions'] = {}
                                else:
                                    st.session_state.bulk_wizard_data['google_suggestions'] = {}
                                
                                # Order results
                                google_suggestions = st.session_state.bulk_wizard_data.get('google_suggestions', {})
                                google_matched = {k: v for k, v in unmatched_cities.items() if google_suggestions.get(k, {}).get('success')}
                                progress_bar.progress(100)
                                status_text.empty()
                                fuzzy_matched = {k: v for k, v in unmatched_cities.items() if fuzzy_results.get(k, {}).get('match_found') and k not in google_matched}
                                no_match = {k: v for k, v in unmatched_cities.items() if k not in google_matched and k not in fuzzy_matched}
                                
                                ordered_unmatched = {**google_matched, **fuzzy_matched, **no_match}
                                st.session_state.bulk_wizard_data['unmatched_cities'] = ordered_unmatched
                                st.session_state.bulk_wizard_data['match_counts'] = {'google': len(google_matched), 'fuzzy': len(fuzzy_matched), 'none': len(no_match)}
                                
                                st.session_state.bulk_wizard_step = 1
                            else:
                                st.session_state.bulk_wizard_step = 2
                            st.rerun()
                        
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    

    # ============================================
    # STEP 1: City Resolution
    # ============================================
    # ============================================
    # STEP 1: City Resolution
    # ============================================
    elif current_step == 1:
        st.markdown("<h3 style='text-align: center;'>Step 2: Resolve Unmatched Cities</h3>", unsafe_allow_html=True)
        
        # 1. LOAD DATA FRESH FROM STATE
        wizard_data = st.session_state.bulk_wizard_data
        unmatched = wizard_data.get('unmatched_cities', {})
        fuzzy_results = wizard_data.get('fuzzy_results', {})
        
        # Ensure 'google_suggestions' is initialized in wizard_data
        if 'google_suggestions' not in wizard_data:
            wizard_data['google_suggestions'] = {}
        google_suggestions = wizard_data['google_suggestions']

        # Ensure 'city_resolutions' exists in session_state
        if 'city_resolutions' not in st.session_state:
            st.session_state.city_resolutions = {}
        
        # Show last distance flush result if any
        if 'last_distance_flush' in st.session_state:
            st.caption(st.session_state['last_distance_flush'])
        
        # Check if we are done
        if not unmatched:
            st.success("All cities matched! Proceeding to distance review...")
            st.session_state.bulk_wizard_step = 2
            st.rerun()

        # 2. CALCULATE LIVE STATS (Before rendering dashboard)
        resolved_count = len(st.session_state.city_resolutions)
        total_count = len(unmatched)
        remaining_count = total_count - resolved_count
        
        # Breakdown of PENDING items only
        pending_google = 0
        pending_fuzzy = 0
        pending_none = 0
        
        for city in unmatched:
            if city not in st.session_state.city_resolutions:
                # Check match status
                has_google = google_suggestions.get(city, {}).get('success', False)
                has_fuzzy = fuzzy_results.get(city, {}).get('match_found', False)
                
                if has_google:
                    pending_google += 1
                elif has_fuzzy:
                    pending_fuzzy += 1
                else:
                    pending_none += 1

        # 3. RENDER DASHBOARD
        st.markdown("""
        <style>
            .stat-box { text-align: center; padding: 10px; border-radius: 8px; border: 1px solid #f0f2f6; background-color: white; }
        </style>
        """, unsafe_allow_html=True)

        c_main, c_breakdown = st.columns([1, 2])
        
        with c_main:
             st.markdown(
                f"""
                <div style="text-align: center; border: 2px solid #ff4b4b; padding: 15px; border-radius: 10px; background-color: #fff5f5;">
                    <h1 style="margin: 0; color: #ff4b4b; font-size: 3rem;">{remaining_count}</h1>
                    <p style="margin: 0; font-weight: bold; color: #555;">Cities Pending</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #888;">{resolved_count} / {total_count} Resolved</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with c_breakdown:
            st.caption("Breakdown of Pending Cities:")
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.metric("🌍 Maps Found", pending_google, help="Google Maps found a location for these")
            with bc2:
                st.metric("🔤 Fuzzy Match", pending_fuzzy, help="Similar spelling found in database")
            with bc3:
                st.metric("❌ No Match", pending_none, help="Requires manual entry or ignore")
            
            # REFRESH BUTTON
            if st.button("🔄 Check Google Maps Again", use_container_width=True, help="Click if waiting for background geocoding"):
                cities_to_check = [c for c in unmatched.keys()]
                if cities_to_check:
                    with st.spinner("Syncing with Google Sheets..."):
                        # Force fresh read
                        new_results = read_geocode_results(cities_to_check)
                        # Update the nested dictionary explicitly
                        st.session_state.bulk_wizard_data['google_suggestions'] = new_results
                        
                        # Calculate how many found for toast
                        found_cnt = sum(1 for c in cities_to_check if new_results.get(c, {}).get('success'))
                        st.toast(f"Sync Complete. Found {found_cnt} locations.", icon="🌍")
                        
                        import time
                        time.sleep(1) # Short pause to ensure state propagates
                        st.rerun()

        st.markdown("---")

        # 4. RESOLUTION LIST
        st.markdown("#### Action Required")
        
        # Sort logic: Google -> Fuzzy -> None -> Resolved
        def sort_priority(c_name):
            if c_name in st.session_state.city_resolutions: return 4
            if google_suggestions.get(c_name, {}).get('success'): return 1
            if fuzzy_results.get(c_name, {}).get('match_found'): return 2
            return 3
            
        sorted_cities = sorted(unmatched.keys(), key=sort_priority)
        
        for idx, city_name in enumerate(sorted_cities):
            info = unmatched[city_name]
            
            # Determine match type
            has_google = google_suggestions.get(city_name, {}).get('success', False)
            has_fuzzy = fuzzy_results.get(city_name, {}).get('match_found', False)
            
            if has_google:
                match_icon = "🌍"
                match_label = "Google Match"
                expanded_default = False 
            elif has_fuzzy:
                match_icon = "🔤"
                match_label = "Fuzzy Match"
                expanded_default = False 
            else:
                match_icon = "❌"
                match_label = "No Match"
                expanded_default = True 
            
            # Check resolution status
            is_resolved = city_name in st.session_state.city_resolutions
            status_icon = "✅" if is_resolved else match_icon
            
            # Collapse if resolved
            is_expanded = False if is_resolved else expanded_default

            with st.expander(f"{status_icon} **{city_name}** ({match_label})", expanded=is_expanded):
                fuzzy = fuzzy_results.get(city_name, {})
                
                # Layout: 2 Columns
                c1, c2 = st.columns([1, 1])
                
                with c1:
                    st.caption(f"Appears in **{len(info['rows'])} routes** (Column: {info['col']})")
                    
                    # 1. Google Match Info
                    google_res = google_suggestions.get(city_name, {})
                    if google_res.get('success'):
                        st.success(f"🌍 **Google Found:** {google_res.get('matched_name')}")
                        st.caption(f"📍 {google_res.get('latitude'):.4f}, {google_res.get('longitude'):.4f}")
                        
                        if st.button("Accept Google Match", key=f"btn_goog_{city_name}", use_container_width=True):
                            prov, reg = get_province_from_coordinates(google_res['latitude'], google_res['longitude'])
                            if not prov: reg = "Central"
                            
                            st.session_state.city_resolutions[city_name] = {
                                'type': 'google_match',
                                'canonical': google_res['matched_name'],
                                'latitude': google_res['latitude'],
                                'longitude': google_res['longitude'],
                                'province': prov,
                                'region': reg,
                                'original_input': city_name
                            }
                            st.rerun()

                    # 2. Fuzzy Match Info
                    elif fuzzy.get('match_found'):
                        st.info(f"🔤 **Similar to:** {fuzzy.get('suggested_canonical')} ({fuzzy.get('confidence')}%)")
                        
                        if st.button(f"Accept '{to_english_city(fuzzy.get('suggested_canonical'))}'", key=f"btn_fuz_{city_name}", use_container_width=True):
                            st.session_state.city_resolutions[city_name] = {
                                'type': 'fuzzy_match',
                                'canonical': fuzzy.get('suggested_canonical'),
                                'confidence': fuzzy.get('confidence'),
                                'english': to_english_city(fuzzy.get('suggested_canonical'))
                            }
                            st.rerun()
                    
                    else:
                        st.warning("No automatic match found.")

                with c2:
                    # Manual / Ignore Options
                    # default index 0 (Input Coordinates)
                    action_mode = st.radio("Manual Action:", ["Input City Coordinates", "Ignore (Skip Routes)"], key=f"rad_{city_name}")
                    
                    if action_mode == "Ignore (Skip Routes)":
                        st.warning("Routes with this city will be excluded.")
                        if st.button("Confirm Ignore", key=f"ign_{city_name}", use_container_width=True):
                             st.session_state.city_resolutions[city_name] = {
                                'type': 'ignored',
                                'rows': info['rows']
                            }
                             st.rerun()
                             
                    elif action_mode == "Input City Coordinates":
                        # st.markdown("**Enter Coordinates:**")
                        col_lat, col_lon = st.columns(2)
                        with col_lat:
                            new_lat = st.number_input("Latitude", key=f"lat_{city_name}", format="%.6f")
                        with col_lon:
                            new_lon = st.number_input("Longitude", key=f"lon_{city_name}", format="%.6f")
                        
                        # Reactive Detection
                        det_prov, det_reg = None, None
                        if new_lat != 0 and new_lon != 0:
                            det_prov, det_reg = get_province_from_coordinates(new_lat, new_lon)
                        
                        # Display Detected Province (Read-Only)
                        if det_prov:
                            st.success(f"📍 Location: {det_prov}")
                        elif new_lat != 0 or new_lon != 0:
                            st.warning("⚠️ No province detected. Please check coordinates.")
                        
                        if st.button("Save New City", key=f"save_{city_name}", use_container_width=True):
                            if new_lat == 0 or new_lon == 0:
                                st.error("Enter valid coordinates")
                            elif not det_prov:
                                st.error("Cannot determine province. Coordinates might be outside covered regions.")
                            else:
                                # Store resolution (logging happens in Continue button)
                                st.session_state.city_resolutions[city_name] = {
                                    'type': 'new_city',
                                    'canonical': city_name,
                                    'latitude': new_lat,
                                    'longitude': new_lon,
                                    'province': det_prov,
                                    'region': det_reg
                                }
                                st.toast(f"✅ Saved: {city_name} → {det_prov}")
                                st.rerun()
        
        # 5. NAVIGATION WITH ADMIN ACTIONS
        st.markdown("---")
        # UPDATED: 4 Columns for Back | Admin Auto | Ignore No-Match | Continue
        c_back, c_auto, c_ignore, c_cont = st.columns([1, 1, 1, 1])
        
        with c_back:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.bulk_wizard_step = 0
                st.rerun()

        with c_auto:
            # NEW: Admin Auto-Resolve Button
            matchable_count = pending_google + pending_fuzzy
            if matchable_count > 0:
                if st.button(f"⚡ Admin Auto-Resolve", use_container_width=True, help=f"Instantly accept {pending_google} Google matches and {pending_fuzzy} Fuzzy matches"):
                    accepted_count = 0
                    for city_name in unmatched.keys():
                        if city_name in st.session_state.city_resolutions:
                            continue
                            
                        # Check Google First
                        google_res = google_suggestions.get(city_name, {})
                        if google_res.get('success'):
                            prov, reg = get_province_from_coordinates(google_res['latitude'], google_res['longitude'])
                            if not prov: reg = "Central"
                            
                            st.session_state.city_resolutions[city_name] = {
                                'type': 'google_match',
                                'canonical': google_res['matched_name'],
                                'latitude': google_res['latitude'],
                                'longitude': google_res['longitude'],
                                'province': prov,
                                'region': reg,
                                'original_input': city_name
                            }
                            accepted_count += 1
                            continue
                            
                        # Check Fuzzy Second
                        fuzzy = fuzzy_results.get(city_name, {})
                        if fuzzy.get('match_found'):
                            st.session_state.city_resolutions[city_name] = {
                                'type': 'fuzzy_match',
                                'canonical': fuzzy.get('suggested_canonical'),
                                'confidence': fuzzy.get('confidence'),
                                'english': to_english_city(fuzzy.get('suggested_canonical'))
                            }
                            accepted_count += 1
                    
                    st.toast(f"✅ Auto-resolved {accepted_count} cities!", icon="⚡")
                    import time
                    time.sleep(1)
                    st.rerun()
            else:
                 st.button("⚡ Admin Auto-Resolve", use_container_width=True, disabled=True)

        with c_ignore:
            # "Ignore No-Matches" logic (Smart Ignore)
            if pending_none > 0:
                if st.button(f"🗑️ Ignore {pending_none} No-Matches", use_container_width=True, help="Ignore only cities with NO Google or Fuzzy matches"):
                    count_ignored = 0
                    for city in unmatched.keys():
                        if city in st.session_state.city_resolutions:
                            continue
                        
                        has_google = google_suggestions.get(city, {}).get('success', False)
                        has_fuzzy = fuzzy_results.get(city, {}).get('match_found', False)
                        
                        if not has_google and not has_fuzzy:
                            st.session_state.city_resolutions[city] = {
                                'type': 'ignored',
                                'rows': unmatched[city]['rows']
                            }
                            count_ignored += 1
                    
                    st.toast(f"Marked {count_ignored} cities as ignored.", icon="🗑️")
                    import time
                    time.sleep(1)
                    st.rerun()
            else:
                 st.button("🗑️ Ignore No-Matches", use_container_width=True, disabled=True)
        
        with c_cont:
            can_proceed = resolved_count >= total_count
            btn_text = "Continue ▶️" if can_proceed else "Resolve All 🚫"
            
            if st.button(btn_text, type="primary", use_container_width=True, disabled=not can_proceed):
                if not can_proceed:
                    st.error("Please resolve or ignore all unmatched cities before proceeding")
                else:
                    # =========================================
                    # APPLY RESOLUTIONS LOGIC
                    # =========================================
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("Applying your resolutions...")
                    resolutions = st.session_state.city_resolutions
                    new_entries = []
                    ignored_rows = set()
                    
                    username = wizard_data.get('username', 'Unknown')

                    for city_name, resolution in resolutions.items():
                        if resolution['type'] == 'ignored':
                            for row_num in resolution.get('rows', []):
                                ignored_rows.add(row_num)
                                # Log exception
                                try:
                                    row_data = wizard_data['parsed_rows'][row_num - 1]
                                    p_val = row_data.get('pickup_raw', '')
                                    d_val = row_data.get('dest_raw', '')
                                except: p_val, d_val = 'Unknown', 'Unknown'
                                
                                log_exception('route_ignored', {
                                    'pickup_city': p_val, 'destination_city': d_val,
                                    'original_value': city_name, 'action': 'Ignored', 'row_num': row_num
                                }, immediate=False)
                                
                        elif resolution['type'] == 'fuzzy_match':
                            log_city_match(city_name, resolution['canonical'], 'Fuzzy Match', resolution['confidence'], user=username, immediate=False)
                            new_entries.append({'variant': city_name, 'canonical': resolution['canonical'], 'region': get_city_region(resolution['canonical']), 'province': get_city_province(resolution['canonical'])})
                            log_to_append_sheet(city_name, resolution['canonical'], get_city_region(resolution['canonical']), province=get_city_province(resolution['canonical']), source='fuzzy', user=username, immediate=False)
                            
                        elif resolution['type'] in ['new_city', 'google_match']:
                            # Use correct match type for each
                            if resolution['type'] == 'google_match':
                                match_type = 'Google Match'
                                src = 'google'
                            else:
                                match_type = 'New City'
                                src = 'new_city'
                            
                            log_city_match(city_name, resolution['canonical'], match_type, None, latitude=resolution['latitude'], longitude=resolution['longitude'], province=resolution['province'], region=resolution['region'], user=username, source=src, immediate=False)
                            
                            new_entries.append({
                                'variant': city_name, 'canonical': resolution['canonical'],
                                'region': resolution['region'], 'province': resolution['province'],
                                'latitude': resolution['latitude'], 'longitude': resolution['longitude']
                            })
                            if resolution['canonical'] != city_name:
                                 new_entries.append({
                                    'variant': resolution['canonical'], 'canonical': resolution['canonical'],
                                    'region': resolution['region'], 'province': resolution['province'],
                                    'latitude': resolution['latitude'], 'longitude': resolution['longitude']
                                })
                            log_to_append_sheet(city_name, resolution['canonical'], resolution['region'], province=resolution['province'], latitude=resolution['latitude'], longitude=resolution['longitude'], source=src, user=username, immediate=False)

                    # Update & Flush
                    progress_bar.progress(60)
                    status_text.text("Updating database and logs...")
                    update_city_normalization_pickle(new_entries)
                    flush_matched_cities_to_sheet()
                    flush_append_sheet()
                    flush_error_log_to_sheet()
                    
                    # Update Parsed Rows
                    progress_bar.progress(80)
                    status_text.text("Finalizing trip data...")
                    parsed_rows = wizard_data['parsed_rows']
                    for row in parsed_rows:
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
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    
                    st.session_state.bulk_wizard_data['parsed_rows'] = parsed_rows
                    st.session_state.bulk_wizard_data['ignored_rows'] = ignored_rows
                    
                    st.session_state.bulk_wizard_step = 2
                    st.rerun()
    
    # ============================================
    # STEP 2: Distance Review
    # ============================================
    elif current_step == 2:
        st.markdown("<h3 style='text-align: center;'>Step 3: Review Distances</h3>", unsafe_allow_html=True)
        
        wizard_data = st.session_state.bulk_wizard_data
        parsed_rows = wizard_data.get('parsed_rows', [])
        username = wizard_data.get('username', '')
        ignored_rows = wizard_data.get('ignored_rows', set())
        
        # Filter out ignored rows
        active_rows = [row for row in parsed_rows if not row.get('ignored', False)]
        
        if ignored_rows:
            st.caption(f"ℹ️ {len(ignored_rows)} routes ignored and excluded from pricing.")
        
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
                # Skip same-city routes (pickup == destination)
                if row['pickup_ar'] == row['dest_ar']:
                    continue
                    
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
            
            # If we logged missing distances, wait for Google Maps API to process them
            if flush_count > 0:
                import time
                status_placeholder = st.empty()
                
                for remaining in range(3, 0, -1):
                    status_placeholder.info(f"⏳ Waiting for Google Maps API to calculate {flush_count} distances... {remaining}s")
                    time.sleep(1)
                
                # Read back newly-resolved distances
                newly_resolved = get_resolved_distances_from_sheet()
                newly_added = 0
                
                for item in newly_resolved:
                    key = (item['pickup_ar'], item['dest_ar'])
                    rev_key = (item['dest_ar'], item['pickup_ar'])
                    if key not in DISTANCE_MATRIX:
                        DISTANCE_MATRIX[key] = item['distance_km']
                        DISTANCE_MATRIX[rev_key] = item['distance_km']
                        newly_added += 1
                        
                        # Update city_pairs if this pair was missing
                        if key in city_pairs and city_pairs[key]['distance'] == 0:
                            city_pairs[key]['distance'] = item['distance_km']
                            city_pairs[key]['source'] = 'Google Maps API'
                
                if newly_added > 0:
                    status_placeholder.success(f"✅ Google Maps resolved {newly_added}/{flush_count} distances!")
                else:
                    status_placeholder.warning(f"⚠️ Google Maps API didn't return results yet. You can enter distances manually below.")
            
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
        
        # Separate missing/new distances from existing (considering user edits)
        # Initialize distance_edits if not exists
        if 'distance_edits' not in st.session_state:
            st.session_state.distance_edits = {}
        
        # Process edits from PREVIOUS rerun BEFORE calculating dashboard metrics
        # This ensures the dashboard updates immediately when editing distances
        if 'distance_editor' in st.session_state and distance_results:
            prev_edited_df = st.session_state.distance_editor
            if hasattr(prev_edited_df, 'iterrows'):
                for index, row in prev_edited_df.iterrows():
                    try:
                        p_ar, d_ar = row['Reference'].split('|')
                        key = (p_ar, d_ar)
                        original_info = distance_results.get(key)
                        if original_info:
                            original_val = original_info['distance']
                            new_val = row['Distance (km)']
                            if abs(new_val - original_val) > 0.001:
                                st.session_state.distance_edits[key] = new_val
                            elif key in st.session_state.distance_edits:
                                del st.session_state.distance_edits[key]
                    except:
                        pass
        
        missing_distances = {}
        existing_distances = {}
        for k, v in distance_results.items():
            # Use edited value if exists, else original
            current_dist = st.session_state.distance_edits.get(k, v['distance'])
            if current_dist == 0 or (current_dist == v['distance'] and v['source'] == 'Missing'):
                missing_distances[k] = v
            else:
                existing_distances[k] = v
        
        # UI: Dashboard Stats - Aligned with bottom navigation [1, 1, 1], centered
        # CSS to center metrics within their columns
        st.markdown("""
        <style>
            [data-testid="stMetric"] {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }
            [data-testid="stMetricLabel"] {
                justify-content: center;
            }
            [data-testid="stMetricValue"] {
                justify-content: center;
            }
        </style>
        """, unsafe_allow_html=True)
        
        c_stat1, c_stat2, c_refresh = st.columns([1, 1, 1])
        
        # Use placeholders for metrics - will be filled AFTER edit processing
        metric_placeholder_1 = c_stat1.empty()
        metric_placeholder_2 = c_stat2.empty()
        
        with c_refresh:
            st.markdown("<br>", unsafe_allow_html=True)  # Align with metrics
            if st.button("🔄 Refresh from Sheets", use_container_width=True, help="Re-check Google Sheets for newly resolved distances"):
                # Clear distance results to force re-check
                if 'distance_results' in st.session_state.bulk_wizard_data:
                    del st.session_state.bulk_wizard_data['distance_results']
                st.rerun()

        st.markdown("---")
        
        # (Edit processing moved to before dashboard calculation - see above)
        
        if distance_results:
            # Prepare data for Data Editor
            editor_data = []
            for pair_key, info in distance_results.items():
                is_missing = info['distance'] == 0 or info['source'] == 'Missing'
                pickup_ar, dest_ar = pair_key
                ref_key = f"{pickup_ar}|{dest_ar}"
                
                # Use current edit if exists, else original
                current_val = st.session_state.distance_edits.get(pair_key, info['distance'])
                
                # Status should reflect CURRENT distance (including user edits)
                is_currently_missing = current_val == 0
                
                editor_data.append({
                    "Reference": ref_key,
                    "Status": "⚠️ Missing" if is_currently_missing else "✅ Ready",
                    "Pickup": info['pickup_en'],
                    "Destination": info['dest_en'],
                    "Distance (km)": float(current_val),
                    "Source": info['source'] if not st.session_state.distance_edits.get(pair_key) else "User Override",
                    "Routes": len(info['rows']),
                    "is_missing_sort": 0 if is_missing else 1 # Hidden sort key
                })
            
            # Sort: Missing first, then by source priority (Google Maps, Fuzzy, others), then by distance
            editor_df = pd.DataFrame(editor_data)
            
            # Create source priority: Missing=0, Google Maps=1, Fuzzy/Historical=2, Others=3
            def source_priority(src):
                if 'Missing' in str(src): return 0
                if 'Google' in str(src): return 1
                if 'Fuzzy' in str(src) or 'Historical' in str(src): return 2
                return 3
            
            editor_df['source_priority'] = editor_df['Source'].apply(source_priority)
            editor_df = editor_df.sort_values(
                by=['is_missing_sort', 'source_priority', 'Distance (km)'],
                ascending=[True, True, True]
            ).drop(columns=['source_priority'])
            
            st.markdown("##### 📝 Review & Edit Distances")
            
            edited_df = st.data_editor(
                editor_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Reference": None,
                    "is_missing_sort": None,
                    "Status": st.column_config.TextColumn(width="small", disabled=True),
                    "Pickup": st.column_config.TextColumn(width="medium", disabled=True),
                    "Destination": st.column_config.TextColumn(width="medium", disabled=True),
                    "Distance (km)": st.column_config.NumberColumn(
                        min_value=0, max_value=5000, step=10, required=True, format="%.1f km"
                    ),
                    "Source": st.column_config.TextColumn(width="medium", disabled=True),
                    "Routes": st.column_config.NumberColumn(width="small", disabled=True)
                },
                key="distance_editor"
            )
            
            # Process edits
            for index, row in edited_df.iterrows():
                try:
                    p_ar, d_ar = row['Reference'].split('|')
                    key = (p_ar, d_ar)
                    
                    # Check if distance changed from ORIGINAL
                    original_info = distance_results.get(key)
                    if original_info:
                        original_val = original_info['distance']
                        new_val = row['Distance (km)']
                        
                        # If different from original, store as edit
                        if abs(new_val - original_val) > 0.001:
                            if st.session_state.distance_edits.get(key) != new_val:
                                st.session_state.distance_edits[key] = new_val
                                st.session_state.distance_edit_changed = True
                        elif key in st.session_state.distance_edits:
                            # Reverted to original
                            del st.session_state.distance_edits[key]
                            st.session_state.distance_edit_changed = True
                except:
                    pass
            
            # Trigger rerun to update Status column and dashboard
            if st.session_state.get('distance_edit_changed'):
                del st.session_state.distance_edit_changed
                st.rerun()
        
        # Recalculate missing based on edits
        missing_count = 0
        for pair_key, info in distance_results.items():
            current_dist = st.session_state.distance_edits.get(pair_key, info['distance'])
            if current_dist == 0 and not info['user_override']:
                missing_count += 1
        
        ready_count = len(distance_results) - missing_count
        all_resolved = missing_count == 0
        
        # NOW fill in the dashboard metric placeholders with correct counts
        metric_placeholder_1.metric("⚠️ Missing Distances", missing_count)
        metric_placeholder_2.metric("✅ Ready to Price", ready_count)
        
        # Navigation
        st.markdown("---")
        
        # =========================================================
        # FIX START: Symmetrical Button Layout & Sizing
        # =========================================================
        # UPDATED: Symmetrical [1, 1, 1] layout for buttons so they are even
        c_back, c_cont, c_reset = st.columns([1, 1, 1])
        
        with c_back:
            # FIX: Added use_container_width=True
            if st.button("⬅️ Back to Cities", use_container_width=True):
                st.session_state.bulk_wizard_step = 1 if wizard_data.get('unmatched_cities') else 0
                st.rerun()
        
        with c_cont:
            proceed_disabled = bool(missing_distances) and not all_resolved
            btn_text = "Generate Pricing ▶️" if not proceed_disabled else "Resolve Missing 🚫"
            
            # FIX: Added use_container_width=True
            if st.button(btn_text, type="primary", use_container_width=True, disabled=proceed_disabled):
                # =========================================================
                # LOGIC PRESERVED: Processing Distance Edits
                # =========================================================
                # Log distance edits as user suggestions
                for pair_key, new_dist in st.session_state.distance_edits.items():
                    pickup_ar, dest_ar = pair_key
                    info = distance_results.get(pair_key, {})
                    old_dist = info.get('distance', 0)
                    
                    if new_dist != old_dist:
                        suggest_distance_change(
                            pickup_ar, dest_ar, 
                            info.get('pickup_en', to_english_city(pickup_ar)), 
                            info.get('dest_en', to_english_city(dest_ar)),
                            old_dist, new_dist, username
                        )
                
                # Store final distances
                final_distances = {}
                distance_sources = {}
                for pair_key, info in distance_results.items():
                    if pair_key in st.session_state.distance_edits:
                        final_distances[pair_key] = st.session_state.distance_edits[pair_key]
                        distance_sources[pair_key] = 'User Edited'
                    elif info['user_override']:
                        final_distances[pair_key] = info['user_override']
                        distance_sources[pair_key] = 'CSV Provided'
                    else:
                        final_distances[pair_key] = info['distance']
                        distance_sources[pair_key] = info['source']
                
                st.session_state.bulk_wizard_data['final_distances'] = final_distances
                st.session_state.bulk_wizard_data['distance_sources'] = distance_sources
                st.session_state.bulk_wizard_step = 3
                st.rerun()
        
        with c_reset:
            # FIX: Added use_container_width=True
            if st.button("🔄 Reset All", use_container_width=True):
                reset_wizard()
                st.rerun()
        
        # =========================================================
        # FIX END
        # =========================================================
        
    # ============================================
    # STEP 3: Final Pricing
    # ============================================
    elif current_step == 3:
        st.markdown("<h3 style='text-align: center;'>Step 4: Final Pricing</h3>", unsafe_allow_html=True)

        wizard_data = st.session_state.bulk_wizard_data
        parsed_rows = wizard_data.get('parsed_rows', [])
        selected_vehicles = wizard_data.get('selected_vehicles', ['Flatbed Trailer'])
        selected_truck_type = wizard_data.get('selected_truck_type', 'Domestic')
        final_distances = wizard_data.get('final_distances', {})
        username = wizard_data.get('username', '')
        ignored_rows = wizard_data.get('ignored_rows', set())
        
        # Filter out ignored rows
        active_rows = [row for row in parsed_rows if not row.get('ignored', False)]
        
        if ignored_rows:
            st.caption(f"ℹ️ {len(ignored_rows)} routes excluded.")
        
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
                    
                    
                    # Determine vehicles to price
                    # If global selection exists, use it. Otherwise use CSV row vehicle.
                    vehicles_to_price = selected_vehicles
                    if not vehicles_to_price and row.get('vehicle_raw'):
                        vehicles_to_price = [row['vehicle_raw']]
                    
                    # Price for each selected vehicle
                    for v_input in vehicles_to_price:
                        # Normalize vehicle name (could be English or Arabic from CSV)
                        v_ar = to_arabic_vehicle(v_input)
                        
                        # If output is default but input wasn't default, try fuzzy matching as fallback
                        if v_ar == DEFAULT_VEHICLE_AR and v_input not in [DEFAULT_VEHICLE_AR, 'Flatbed Trailer', '']:
                            # Try fuzzy matching using existing RapidFuzz
                            if RAPIDFUZZ_AVAILABLE:
                                candidates = list(VEHICLE_TYPE_EN.keys()) + list(VEHICLE_TYPE_AR.keys())
                                matches = process.extract(str(v_input).strip(), candidates, scorer=fuzz.token_sort_ratio, limit=1)
                                if matches and matches[0][1] >= 60:  # 60% threshold
                                    best_match = matches[0][0]
                                    v_ar = to_arabic_vehicle(best_match)

                        # Skip if normalization failed completely (safe fallback)
                        if not v_ar: v_ar = DEFAULT_VEHICLE_AR
                        
                        result = lookup_route_stats(
                            pickup_ar, dest_ar, v_ar,
                            dist_override=dist_override,
                            truck_type=selected_truck_type
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
            
            st.success(f"✅ Generated pricing for **{len(res_df)}** route-vehicle combinations")
            
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            # Save section
            st.markdown("---")
            
            # Centered Export Options
            col_d1, col_export, col_d2 = st.columns([1, 4, 1])
            with col_export:
                st.markdown("<h4 style='text-align: center;'>💾 Export Results</h4>", unsafe_allow_html=True)
                
                rfq_sheet_url = st.secrets.get('RFQ_url', '')
                
                c_down, c_up = st.columns(2)
                
                with c_down:
                    st.download_button(
                        "📥 Download CSV",
                        res_df.to_csv(index=False),
                        "pricing_results.csv",
                        "text/csv",
                        type="primary",
                        use_container_width=True
                    )
                
                with c_up:
                    if rfq_sheet_url:
                        default_sheet_name = f"Pricing_{datetime.now().strftime('%Y%m%d_%H%M')}"
                        # Using a unique key for the popover or expander to keep UI clean
                        with st.expander("☁️ Upload to Google Sheet"):
                             sheet_name = st.text_input("Sheet Name", value=default_sheet_name, key='result_sheet_name')
                             
                             if st.button("Start Upload", use_container_width=True):
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
            
            st.markdown("---")
            
            # Bottom Nav: Symmetrical
            b1, b2, b3 = st.columns([1, 1, 1])
            
            with b1:
                if st.button("⬅️ Back", key="step3_back_bottom", use_container_width=True):
                    st.session_state.bulk_results_stale = True
                    st.session_state.bulk_wizard_step = 2
                    st.rerun()
            
            with b2:
                 st.empty() # Spacer
                 
            with b3:
                 if st.button("🔄 Start New", key="step3_new_bottom", use_container_width=True):
                    reset_wizard()
                    st.rerun()

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

# --- TAB 4: ADMIN ---
with tab4:
    st.subheader("🔧 Admin Panel")
    
    # Password protection
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.warning("🔒 This section requires admin access.")
        
        admin_password = st.text_input("Enter Admin Password", type="password", key="admin_pwd_input")
        
        if st.button("🔓 Unlock Admin Panel", type="primary"):
            correct_password = st.secrets.get('admin_password', '')
            if admin_password == correct_password and correct_password != '':
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect password")
    else:
        # Show logout option
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔒 Lock", use_container_width=True):
                st.session_state.admin_authenticated = False
                st.rerun()
        
        st.success("✅ Admin access granted")
        st.markdown("---")
        
        # ============================================
        # MASTER GRID GENERATOR
        # ============================================
        st.markdown("### ⚡ Generate Master Grid")
        st.warning("⚠️ This will generate pricing for ALL city combinations (~2,500+ routes). It may take a few minutes.")
        st.caption(f"Results will be uploaded to: {BULK_PRICING_SHEET_URL}")
        
        batch_size = st.slider("Batch size", min_value=100, max_value=500, value=250, step=50, 
                               help="Number of routes to generate and upload at a time", key="admin_batch_size")
        
        # Check if there's a previous incomplete run
        has_incomplete = 'master_grid_progress' in st.session_state and st.session_state.master_grid_progress.get('incomplete', False)
        
        if has_incomplete:
            prog = st.session_state.master_grid_progress
            
            if prog.get('error_message'):
                st.error(f"❌ {prog['error_message']}")
            
            st.warning(f"⚠️ Previous run incomplete: {prog['rows_written']:,}/{prog['total_routes']:,} routes uploaded.")
            
            col1, col2 = st.columns(2)
            with col1:
                resume_clicked = st.button("🔄 Resume Upload", type="primary", key="admin_resume")
            with col2:
                if st.button("🗑️ Start Fresh", key="admin_fresh"):
                    del st.session_state.master_grid_progress
                    if 'master_grid_df' in st.session_state:
                        del st.session_state.master_grid_df
                    st.rerun()
        else:
            resume_clicked = False
        
        start_fresh_clicked = st.button("🚀 Run & Upload Master Grid", key="admin_run") if not has_incomplete else False
        
        if start_fresh_clicked or resume_clicked:
            import time
            
            wks = get_bulk_sheet()
            if not wks:
                st.error("❌ Google Cloud credentials not found or sheet access denied.")
            else:
                combos = [(p, d) for p, d in itertools.product(all_canonicals, all_canonicals) if p != d]
                total_routes = len(combos)
                
                ensure_sheet_rows(wks, total_routes + 10)
                
                if resume_clicked and has_incomplete:
                    prog = st.session_state.master_grid_progress
                    start_index = prog['next_index']
                    rows_written = prog['rows_written']
                    batch_num = prog['batch_num']
                    all_results = prog.get('all_results', [])
                    st.info(f"📊 Resuming from route {start_index:,} ({rows_written:,} already uploaded)...")
                else:
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
                
                for i in range(start_index, total_routes, batch_size):
                    batch_combos = combos[i:i + batch_size]
                    batch_num += 1
                    
                    status_text.text(f"Batch {batch_num}: Generating {len(batch_combos)} routes...")
                    batch_results = []
                    for p_ar, d_ar in batch_combos:
                        batch_results.append(lookup_route_stats(p_ar, d_ar, DEFAULT_VEHICLE_AR, check_history=True))
                    
                    all_results.extend(batch_results)
                    
                    pending_errors = len(st.session_state.get('error_log_pending', []))
                    pending_distances = len(st.session_state.get('matched_distances_pending', []))
                    if pending_errors >= 500:
                        flush_error_log_to_sheet(batch_size=500)
                    if pending_distances >= 100:
                        flush_matched_distances_to_sheet()
                    
                    status_text.text(f"Batch {batch_num}: Writing {len(batch_results)} rows to sheet...")
                    try:
                        start_row = rows_written + 2
                        batch_df = pd.DataFrame(batch_results)
                        data_rows = batch_df.astype(str).values.tolist()
                        wks.update(f'A{start_row}', data_rows)
                        rows_written += len(batch_results)
                    except Exception as e:
                        st.session_state.master_grid_progress = {
                            'incomplete': True,
                            'next_index': i + batch_size,
                            'rows_written': rows_written,
                            'batch_num': batch_num,
                            'total_routes': total_routes,
                            'all_results': all_results,
                            'error_message': f"Failed at batch {batch_num}: {str(e)}"
                        }
                        flush_error_log_to_sheet(batch_size=500)
                        flush_matched_distances_to_sheet()
                        st.rerun()
                    
                    progress_bar.progress(min(1.0, (i + len(batch_combos)) / total_routes))
                    status_text.text(f"✓ Batch {batch_num} complete | {rows_written:,}/{total_routes:,} routes uploaded")
                    
                    if i + batch_size < total_routes:
                        time.sleep(0.3)
                
                try:
                    wks.update(f'A{rows_written + 2}', [[f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]])
                except:
                    pass
                
                progress_bar.progress(1.0)
                st.success(f"✅ Successfully uploaded {rows_written:,} routes in {batch_num} batches!")
                status_text.empty()
                
                if 'master_grid_progress' in st.session_state:
                    del st.session_state.master_grid_progress
                
                flushed_ok, flushed_count = flush_error_log_to_sheet(batch_size=500)
                if flushed_count > 0:
                    st.caption(f"📝 Logged {flushed_count} exceptions to error sheet")
                
                dist_ok, dist_count = flush_matched_distances_to_sheet()
                if dist_count > 0:
                    st.caption(f"📏 Logged {dist_count} missing distances for Google Maps lookup")
                
                st.session_state.master_grid_df = pd.DataFrame(all_results)
        
        if 'master_grid_df' in st.session_state:
            st.download_button(
                "📥 Download Master Grid CSV", 
                st.session_state.master_grid_df.to_csv(index=False), 
                "master_grid.csv", 
                "text/csv",
                key="admin_download_grid"
            )
        
        st.markdown("---")
        
        # ============================================
        # DISTANCE UPDATE FROM GOOGLE SHEETS
        # ============================================
        st.markdown("### 📏 Update Distances from Google Sheets")
        
        resolved = get_resolved_distances_from_sheet()
        failed = get_failed_distances_from_sheet()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Ready to Import", len(resolved))
        col2.metric("Failed Lookups", len(failed))
        col3.metric("Pickle Size", f"{len(DISTANCE_MATRIX):,}")
        
        if resolved or failed:
            t1, t2 = st.tabs(["✅ Ready to Import", "⚠️ Failed Lookups"])
            
            with t1:
                if resolved:
                    preview_df = pd.DataFrame([
                        {'From': to_english_city(r['pickup_ar']), 
                         'To': to_english_city(r['dest_ar']), 
                         'Distance': f"{r['distance_km']} km",
                         'Source': '✏️ User' if r.get('is_suggestion') else '🌐 API'}
                        for r in resolved[:20]
                    ])
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
                    if len(resolved) > 20:
                        st.caption(f"... and {len(resolved) - 20} more")
                else:
                    st.info("No pending distances to import.")

            with t2:
                if failed:
                    failed_df = pd.DataFrame([
                        {'From': f['pickup_en'] or to_english_city(f['pickup_ar']),
                         'To': f['dest_en'] or to_english_city(f['dest_ar']),
                         'Error': f['error_value']}
                        for f in failed[:20]
                    ])
                    st.dataframe(failed_df, use_container_width=True, hide_index=True)
                    if len(failed) > 20:
                        st.caption(f"... and {len(failed) - 20} more")
                else:
                    st.success("No failed lookups!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Refresh", use_container_width=True, key="admin_refresh"):
                st.rerun()

        with col2:
            if st.button("🔄 Import to Pickle", type="primary", disabled=len(resolved)==0, 
                        use_container_width=True, key="admin_import"):
                with st.spinner("Importing distances..."):
                    success, count, message = update_distance_pickle_from_sheet()
                    if success:
                        st.success(message)
                        if count > 0:
                            st.balloons()
                            st.cache_resource.clear()
                    else:
                        st.error(message)
            
        with col3:
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
                    key="admin_download_pkl"
                )

st.caption("Freight Pricing Tool | Buy Price rounded to 100 | Sell Price rounded to 50 | All prices in SAR")
