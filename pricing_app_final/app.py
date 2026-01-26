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
from rapidfuzz import process, fuzz  # --- NEW: Added for fuzzy matching ---

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
    """
    if pd.isna(text) or text is None:
        return None
    text = str(text).strip().lower()
    # Remove all non-alphanumeric characters (keeps Arabic letters too)
    text = re.sub(r'[^a-z0-9\u0600-\u06ff]', '', text)
    return text if text else None

# ============================================
# ERROR LOGGING (Google Sheets)
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

# --- NEW: VARIANTS SHEET HELPER ---
@st.cache_resource
def get_variants_sheet():
    """Get or create the Variants sheet for approved fuzzy matches."""
    try:
        client = get_gsheet_client()
        if client is None: return None
        
        sheet_url = st.secrets.get('error_log_sheet_url')
        if not sheet_url: return None
        
        spreadsheet = client.open_by_url(sheet_url)
        
        try:
            worksheet = spreadsheet.worksheet('Variants')
        except:
            worksheet = spreadsheet.add_worksheet(title='Variants', rows=1000, cols=5)
            worksheet.update('A1:E1', [['Timestamp', 'Original_Variant', 'Mapped_Canonical', 'Confidence_Score', 'User_Approved']])
        
        return worksheet
    except Exception as e:
        return None

def log_variants_to_sheet(variants_list):
    """Log approved variants to the Google Sheet."""
    try:
        worksheet = get_variants_sheet()
        if not worksheet: return False
        
        rows = []
        for v in variants_list:
            rows.append([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                v['variant'],
                v['canonical'],
                str(v['score']),
                'Yes'
            ])
        
        worksheet.append_rows(rows)
        return True
    except Exception as e:
        return False
# ----------------------------------

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

def flush_matched_distances_to_sheet():
    """
    Flush pending matched distances to Google Sheet.
    Call at the end of bulk operations.
    """
    if 'matched_distances_pending' not in st.session_state or len(st.session_state.matched_distances_pending) == 0:
        return True, 0
    
    pending = st.session_state.matched_distances_pending
    
    try:
        worksheet = get_matched_distances_sheet()
        if not worksheet:
            return False, 0
        
        # Get existing data to check for duplicates and find next row
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
            
            # Add to existing pairs to avoid duplicates within this batch
            existing_pairs.add((item['pickup_ar'], item['dest_ar']))
            existing_pairs.add((item['dest_ar'], item['pickup_ar']))
            next_row += 1
        
        if rows_to_add:
            # Write all rows at once
            start_row = len(existing) + 1
            worksheet.update(f'A{start_row}:I{start_row + len(rows_to_add) - 1}', rows_to_add, value_input_option='USER_ENTERED')
        
        # Clear pending queue
        st.session_state.matched_distances_pending = []
        return True, len(rows_to_add)
    
    except Exception as e:
        return False, 0

def get_resolved_distances_from_sheet():
    """
    Read resolved distances from the MatchedDistances sheet.
    Returns list of dicts with pickup_ar, dest_ar, distance for rows with valid distances.
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
    """Read FAILED distance lookups from the MatchedDistances sheet."""
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
            # Need at least 5 columns
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
                    try:
                        dist_clean = dist_str.replace(',', '').replace(' km', '').replace('km', '').strip()
                        dist_km = float(dist_clean)
                        if dist_km <= 0:
                            is_failed = True
                            error_type = f'Invalid ({dist_km})'
                    except (ValueError, TypeError):
                        # Not a valid number
                        if dist_str:
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
# BULK ADJUSTMENTS SHEET
# ============================================
def get_bulk_adjustments_sheet():
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
        return False, 0
    
    return True, 0

# ============================================
# BULK SHEET HELPER
# ============================================
def get_bulk_sheet():
    try:
        client = get_gsheet_client()
        if client is None:
            return None
        
        spreadsheet = client.open_by_url(BULK_PRICING_SHEET_URL)
        
        try:
            worksheet = spreadsheet.worksheet(BULK_TAB_NAME)
        except:
            # Create the sheet if it doesn't exist
            worksheet = spreadsheet.add_worksheet(title=BULK_TAB_NAME, rows=10000, cols=20)
            
        return worksheet
    except Exception as e:
        st.error(f"Google Sheets connection error: {str(e)}")
        return None

def ensure_sheet_rows(worksheet, required_rows):
    try:
        current_rows = worksheet.row_count
        if current_rows < required_rows:
            new_rows = int(required_rows * 1.2)
            worksheet.resize(rows=new_rows)
            return True
    except Exception as e:
        pass
    return False

def log_exception(exception_type, details, immediate=False):
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
        try:
            worksheet = get_error_sheet()
            if worksheet:
                worksheet.append_row(row)
        except Exception as e:
            pass
    else:
        st.session_state.error_log_pending.append(row)

def flush_error_log_to_sheet(batch_size=500):
    if 'error_log_pending' not in st.session_state or len(st.session_state.error_log_pending) == 0:
        return True, 0
    
    pending = st.session_state.error_log_pending
    total_count = len(pending)
    
    try:
        worksheet = get_error_sheet()
        if worksheet:
            # Write in batches
            written = 0
            for i in range(0, total_count, batch_size):
                batch = pending[i:i + batch_size]
                worksheet.append_rows(batch)
                written += len(batch)
            
            st.session_state.error_log_pending = []
            return True, written
    except Exception as e:
        return False, 0
    
    return True, 0

def report_distance_issue(pickup_ar, dest_ar, pickup_en, dest_en, current_distance, issue_type, user_notes=''):
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
    success_reported = False
    success_matched = False
    
    # 1. Log to Reported sheet
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
    
    # 2. Log to MatchedDistances sheet
    try:
        worksheet = get_matched_distances_sheet()
        if worksheet:
            existing = worksheet.get_all_values()
            row_to_update = None
            for i, row in enumerate(existing[1:], start=2):
                if len(row) >= 3:
                    if (row[1] == pickup_ar and row[2] == dest_ar) or \
                       (row[1] == dest_ar and row[2] == pickup_ar):
                        row_to_update = i
                        break
            
            dist_int = int(round(suggested_distance))
            
            if row_to_update:
                worksheet.update(f'F{row_to_update}:I{row_to_update}', [[
                    '', str(dist_int), f'Suggested by {user_name}', 'No'
                ]])
            else:
                next_row = len(existing) + 1
                row_data = [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    pickup_ar,
                    dest_ar,
                    pickup_en,
                    dest_en,
                    '', str(dist_int), f'Suggested by {user_name}', 'No'
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
    if 'error_log' not in st.session_state or len(st.session_state.error_log) == 0:
        return None
    return pd.DataFrame(st.session_state.error_log).to_csv(index=False)

def clear_error_log():
    st.session_state.error_log = []

def upload_to_gsheet(df, batch_size=500, progress_callback=None):
    import time
    try:
        wks = get_bulk_sheet()
        if not wks:
            return False, "❌ Google Cloud credentials not found or sheet access denied."
        
        try:
            wks.clear()
        except Exception as e:
            return False, f"❌ Failed to clear sheet: {str(e)}"
        
        headers = df.columns.values.tolist()
        data_rows = df.astype(str).values.tolist()
        total_rows = len(data_rows)
        
        try:
            wks.update('A1', [headers])
        except Exception as e:
            return False, f"❌ Failed to write headers: {str(e)}"
        
        rows_written = 0
        batch_num = 0
        
        for i in range(0, total_rows, batch_size):
            batch = data_rows[i:i + batch_size]
            start_row = i + 2
            try:
                range_str = f'A{start_row}'
                wks.update(range_str, batch)
                rows_written += len(batch)
                batch_num += 1
                if progress_callback: progress_callback(rows_written, total_rows, batch_num)
                if i + batch_size < total_rows: time.sleep(0.5)
            except Exception as e:
                return False, f"❌ Failed at batch {batch_num + 1}: {str(e)}"
        
        try:
            wks.update(f'A{total_rows + 2}', [[f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]])
        except: pass
            
        return True, f"✅ Successfully uploaded {rows_written} rows to '{BULK_TAB_NAME}'"
    except Exception as e:
        return False, f"❌ Upload error: {str(e)}"

def upload_to_rfq_sheet(df, sheet_name, batch_size=500, progress_callback=None):
    import time
    try:
        client = get_gsheet_client()
        if client is None: return False, "❌ Google Cloud credentials not found."
        
        rfq_url = st.secrets.get('RFQ_url')
        if not rfq_url: return False, "❌ RFQ_url not configured."
        
        try:
            spreadsheet = client.open_by_url(rfq_url)
        except Exception as e:
            return False, f"❌ Failed to open RFQ spreadsheet: {str(e)}"
        
        existing_sheets = [ws.title for ws in spreadsheet.worksheets()]
        if sheet_name in existing_sheets:
            return False, f"❌ Sheet '{sheet_name}' already exists."
        
        try:
            total_rows = len(df) + 5
            total_cols = len(df.columns) + 2
            wks = spreadsheet.add_worksheet(title=sheet_name, rows=total_rows, cols=total_cols)
        except Exception as e:
            return False, f"❌ Failed to create sheet '{sheet_name}': {str(e)}"
        
        headers = df.columns.values.tolist()
        data_rows = df.astype(str).values.tolist()
        total_data_rows = len(data_rows)
        
        try:
            wks.update('A1', [headers])
        except Exception as e:
            return False, f"❌ Failed to write headers: {str(e)}"
        
        rows_written = 0
        batch_num = 0
        
        for i in range(0, total_data_rows, batch_size):
            batch = data_rows[i:i + batch_size]
            start_row = i + 2
            try:
                range_str = f'A{start_row}'
                wks.update(range_str, batch)
                rows_written += len(batch)
                batch_num += 1
                if progress_callback: progress_callback(rows_written, total_data_rows, batch_num)
                if i + batch_size < total_data_rows: time.sleep(0.5)
            except Exception as e:
                return False, f"❌ Failed at batch {batch_num + 1}: {str(e)}"
        
        try:
            wks.update(f'A{total_data_rows + 2}', [[f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]])
        except: pass
            
        return True, f"✅ Successfully uploaded {rows_written} rows to '{sheet_name}'"
    except Exception as e:
        return False, f"❌ Upload error: {str(e)}"

def populate_rfq_template_sheet(template_df):
    try:
        client = get_gsheet_client()
        if client is None: return False, "❌ Google Cloud credentials not found."
        rfq_url = st.secrets.get('RFQ_url')
        if not rfq_url: return False, "❌ RFQ_url not configured."
        
        try:
            spreadsheet = client.open_by_url(rfq_url)
        except Exception as e:
            return False, f"❌ Failed to open RFQ spreadsheet: {str(e)}"
        
        existing_sheets = [ws.title for ws in spreadsheet.worksheets()]
        if 'Template' in existing_sheets:
            wks = spreadsheet.worksheet('Template')
            wks.clear()
        else:
            wks = spreadsheet.add_worksheet(title='Template', rows=100, cols=10)
        
        headers = template_df.columns.values.tolist()
        data_rows = template_df.astype(str).values.tolist()
        data_rows = [['' if cell == 'nan' else cell for cell in row] for row in data_rows]
        
        all_data = [headers] + data_rows
        wks.update('A1', all_data)
        
        return True, "✅ Template sheet updated successfully!"
    except Exception as e:
        return False, f"❌ Error updating template: {str(e)}"

# ============================================
# FIXED SETTINGS
# ============================================
N_SIMILAR = 10
MIN_DAYS_APART = 5
MAX_AGE_DAYS = 180
RECENT_PRIORITY_DAYS = 30
RECENCY_WINDOW = 90
RENTAL_COST_PER_DAY = 800
RENTAL_KM_PER_DAY = 600
ERROR_BARS = {'High': 0.10, 'Medium': 0.15, 'Low': 0.25, 'Very Low': 0.35}
MARGIN_HIGH_BACKHAUL = 0.12
MARGIN_MEDIUM_BACKHAUL = 0.18
MARGIN_LOW_BACKHAUL = 0.25
MARGIN_UNKNOWN = 0.20
BACKHAUL_HIGH_THRESHOLD = 0.85
BACKHAUL_MEDIUM_THRESHOLD = 0.65

# ============================================
# LOAD CITY NORMALIZATION
# ============================================
@st.cache_resource
def load_city_normalization():
    norm_path = os.path.join(APP_DIR, 'model_export', 'city_normalization_with_regions.csv')
    v_to_c, v_to_c_lower, v_to_c_agg, v_to_reg, c_to_reg, v_to_prov, c_to_prov, c_to_en = {}, {}, {}, {}, {}, {}, {}, {}
    
    if os.path.exists(norm_path):
        try:
            df = pd.read_csv(norm_path)
            for _, row in df.iterrows():
                v, c = str(row['variant']).strip(), str(row['canonical']).strip()
                reg, prov = row.get('region'), row.get('province')
                
                v_to_c[v] = c
                v_lower = normalize_english_text(v)
                if v_lower: v_to_c_lower[v_lower] = c
                v_agg = normalize_aggressive(v)
                if v_agg: v_to_c_agg[v_agg] = c
                
                if pd.notna(reg): 
                    v_to_reg[v] = reg
                    if c not in c_to_reg: c_to_reg[c] = reg
                if pd.notna(prov):
                    v_to_prov[v] = prov
                    if c not in c_to_prov: c_to_prov[c] = prov
            
            grouped = df.groupby('canonical')['variant'].apply(list)
            for c, vars in grouped.items():
                ascii_vars = [v for v in vars if re.match(r'^[A-Za-z\s\-\(\)\.]+$', str(v))]
                if ascii_vars:
                    title = [v for v in ascii_vars if str(v)[0].isupper() and not str(v).isupper()]
                    c_to_en[c] = title[0] if title else ascii_vars[0]
                else: c_to_en[c] = c
        except Exception as e: st.error(f"Error reading normalization file: {e}")
    return v_to_c, v_to_c_lower, v_to_c_agg, v_to_reg, c_to_reg, v_to_prov, c_to_prov, c_to_en

(CITY_TO_CANONICAL, CITY_TO_CANONICAL_LOWER, CITY_TO_CANONICAL_AGGRESSIVE,
 CITY_TO_REGION, CANONICAL_TO_REGION, CITY_TO_PROVINCE, CANONICAL_TO_PROVINCE,
 CITY_AR_TO_EN) = load_city_normalization()
CITY_EN_TO_AR = {v: k for k, v in CITY_AR_TO_EN.items()}

# --- NEW: RAPIDFUZZ MATCH HELPER ---
def find_best_match(city, choices, cutoff=80):
    """Find best fuzzy match using RapidFuzz (Levenshtein). cutoff is 0-100."""
    if not city or not choices: return None, 0.0
    # extractOne returns (match, score, index)
    result = process.extractOne(city, choices, scorer=fuzz.token_sort_ratio, score_cutoff=cutoff)
    if result:
        match, score, _ = result
        return match, round(score / 100, 2)
    return None, 0.0
# -----------------------------------

def normalize_city(city_raw):
    """Normalize city name."""
    if pd.isna(city_raw) or city_raw == '': return None, False
    city = str(city_raw).strip()
    
    # --- CHECK RUNTIME VARIANTS (From user approval in Tab 2) ---
    if 'runtime_variants' in st.session_state and city in st.session_state.runtime_variants:
        return st.session_state.runtime_variants[city], True
    # -----------------------------------------------------------

    if city in CITY_TO_CANONICAL: return CITY_TO_CANONICAL[city], True
    city_lower = normalize_english_text(city)
    if city_lower and city_lower in CITY_TO_CANONICAL_LOWER: return CITY_TO_CANONICAL_LOWER[city_lower], True
    city_agg = normalize_aggressive(city)
    if city_agg and city_agg in CITY_TO_CANONICAL_AGGRESSIVE: return CITY_TO_CANONICAL_AGGRESSIVE[city_agg], True
    if city in CITY_EN_TO_AR: return CITY_EN_TO_AR[city], True
    return city, False

def get_city_region(city):
    if city in CITY_TO_REGION: return CITY_TO_REGION[city]
    if city in CANONICAL_TO_REGION: return CANONICAL_TO_REGION[city]
    canonical = CITY_TO_CANONICAL.get(city, city)
    if canonical in CANONICAL_TO_REGION: return CANONICAL_TO_REGION[canonical]
    return None

def get_city_province(city):
    if city in CITY_TO_PROVINCE: return CITY_TO_PROVINCE[city]
    if city in CANONICAL_TO_PROVINCE: return CANONICAL_TO_PROVINCE[city]
    canonical = CITY_TO_CANONICAL.get(city, city)
    if canonical in CANONICAL_TO_PROVINCE: return CANONICAL_TO_PROVINCE[canonical]
    return None

def classify_unmatched_city(city_raw):
    if pd.isna(city_raw) or str(city_raw).strip() == '':
        return {'error_type': 'empty', 'message': 'Empty city name', 'action': 'Provide a valid city name', 'in_historical_data': False}
    city = str(city_raw).strip()
    in_historical = city in VALID_CITIES_AR
    region = get_city_region(city)
    if in_historical and not region:
        return {'error_type': 'no_region', 'message': f'City "{city}" exists in historical data but has no region', 'action': 'Add to canonicals', 'in_historical_data': True}
    elif not in_historical:
        return {'error_type': 'unknown_city', 'message': f'City "{city}" not found', 'action': 'Check variant', 'in_historical_data': False}
    return {'error_type': 'unknown', 'message': f'City "{city}" - unknown issue', 'action': 'Investigate', 'in_historical_data': in_historical}

def to_english_city(city_ar):
    if city_ar in CITY_AR_TO_EN: return CITY_AR_TO_EN[city_ar]
    norm, found = normalize_city(city_ar)
    if found and norm in CITY_AR_TO_EN: return CITY_AR_TO_EN[norm]
    return city_ar

def to_arabic_city(city_en):
    if city_en in CITY_EN_TO_AR: return CITY_EN_TO_AR[city_en]
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
    'دينا': 'Dynna (Small)',         # Standard 'Dynna' defaults to Small
    'دينا كبيرة': 'Dynna (Large)',   # New explicit type for Large
    'لوري': 'Lorries',
    'Unknown': 'Unknown',
}
VEHICLE_TYPE_AR = {v: k for k, v in VEHICLE_TYPE_EN.items()}
DEFAULT_VEHICLE_AR = 'تريلا فرش'
DEFAULT_VEHICLE_EN = 'Flatbed Trailer'

def to_english_vehicle(vtype_ar): return VEHICLE_TYPE_EN.get(vtype_ar, vtype_ar)
def to_arabic_vehicle(vtype_en):
    if vtype_en in VEHICLE_TYPE_AR: return VEHICLE_TYPE_AR[vtype_en]
    if vtype_en in VEHICLE_TYPE_EN: return vtype_en
    return DEFAULT_VEHICLE_AR

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
def to_english_commodity(c_ar): return COMMODITY_EN.get(c_ar, c_ar)
def to_arabic_commodity(c_en): return COMMODITY_AR.get(c_en, c_en)

# ============================================
# MODELS
# ============================================
class IndexShrinkagePredictor:
    def __init__(self, m):
        self.current_index = m['current_index']
        self.lane_multipliers = m['lane_multipliers']
        self.global_mean = m['global_mean']
        self.k = m['k_prior_strength']
        self.pickup_priors = m['pickup_priors']
        self.dest_priors = m['dest_priors']
        self.lane_stats = m['lane_stats']
        self.city_to_region = m.get('city_to_region', {})
        self.regional_cpk = {tuple(k.split('|')): v for k, v in m.get('regional_cpk', {}).items()} if isinstance(list(m.get('regional_cpk', {}).keys())[0] if m.get('regional_cpk') else '', str) else m.get('regional_cpk', {})
        self.model_date = m.get('model_date', 'Unknown')
    
    def get_canonical_lane(self, lane):
        parts = lane.split(' → ')
        if len(parts) == 2:
            p = parts[0]
            d = parts[1]
            p_can = CITY_TO_CANONICAL.get(p) or CITY_TO_CANONICAL_LOWER.get(normalize_english_text(p), p)
            d_can = CITY_TO_CANONICAL.get(d) or CITY_TO_CANONICAL_LOWER.get(normalize_english_text(d), d)
            return f"{p_can} → {d_can}"
        return lane
    
    def has_lane_data(self, lane):
        if lane in self.lane_stats or lane in self.lane_multipliers: return True
        cl = self.get_canonical_lane(lane)
        return cl in self.lane_stats or cl in self.lane_multipliers
    
    def predict(self, pickup, dest, dist=None):
        lane = f"{pickup} → {dest}"
        cl = self.get_canonical_lane(lane)
        lookup = cl if (cl in self.lane_stats or cl in self.lane_multipliers) else lane
        
        idx_pred = None
        if lookup in self.lane_multipliers:
            idx_pred = self.current_index * self.lane_multipliers[lookup]
        
        p_prior = self.pickup_priors.get(pickup, self.pickup_priors.get(CITY_TO_CANONICAL.get(pickup, pickup), self.global_mean))
        d_prior = self.dest_priors.get(dest, self.dest_priors.get(CITY_TO_CANONICAL.get(dest, dest), self.global_mean))
        city_prior = (p_prior + d_prior) / 2
        
        stats = self.lane_stats.get(lane) or self.lane_stats.get(lookup)
        if stats:
            lam = stats['lane_n'] / (stats['lane_n'] + self.k)
            shrink_pred = lam * stats['lane_mean'] + (1 - lam) * city_prior
        else:
            shrink_pred = city_prior
            
        if idx_pred is not None:
            pred = (idx_pred + shrink_pred) / 2
            method = 'Index + Shrinkage'
            conf = 'High' if stats and stats['lane_n'] >= 20 else 'Medium' if stats else 'Low'
        else:
            pred = shrink_pred
            method = 'Shrinkage'
            conf = 'Medium'
            
        err = ERROR_BARS.get(conf, 0.25)
        return {'predicted_cost': pred, 'cpk_low': pred*(1-err), 'cpk_high': pred*(1+err), 'method': method, 'confidence': conf,
                'predicted_cost': round(pred*dist, 0) if dist else round(pred, 3)}

class BlendPredictor:
    def __init__(self, m):
        self.config = m['config']
        self.province_weight = self.config.get('province_weight', 0.7)
        self.current_index = m['current_index']
        self.pickup_city_mult = m['pickup_city_mult']
        self.dest_city_mult = m['dest_city_mult']
        self.city_to_province = m.get('city_to_province', {})
        self.province_cpk = m.get('province_cpk', {})
        self.city_to_region = m.get('city_to_region', {})
        self.regional_cpk = m.get('regional_cpk', {})
        self.using_provinces = len(set(self.city_to_province.values())) > 5

    def predict(self, pickup, dest, dist=None, pickup_region_override=None, dest_region_override=None):
        p_can = CITY_TO_CANONICAL.get(pickup, pickup)
        d_can = CITY_TO_CANONICAL.get(dest, dest)
        
        p_prov = CANONICAL_TO_PROVINCE.get(p_can) or self.city_to_province.get(p_can) or self.city_to_province.get(pickup)
        d_prov = CANONICAL_TO_PROVINCE.get(d_can) or self.city_to_province.get(d_can) or self.city_to_province.get(dest)
        prov_cpk = self.province_cpk.get((p_prov, d_prov)) if p_prov and d_prov else None
        
        reg_cpk = None
        if prov_cpk is None:
            p_reg = CANONICAL_TO_REGION.get(p_can) or self.city_to_region.get(p_can) or pickup_region_override
            d_reg = CANONICAL_TO_REGION.get(d_can) or self.city_to_region.get(d_can) or dest_region_override
            if p_reg and d_reg: reg_cpk = self.regional_cpk.get((p_reg, d_reg))
            
        p_mult = self.pickup_city_mult.get(pickup, self.pickup_city_mult.get(p_can, 1.0))
        d_mult = self.dest_city_mult.get(dest, self.dest_city_mult.get(d_can, 1.0))
        city_cpk = p_mult * d_mult * self.current_index
        
        if prov_cpk is not None:
            pred = self.province_weight * prov_cpk + (1 - self.province_weight) * city_cpk
            method = 'Blend (Province)'
        elif reg_cpk is not None:
            pred = self.province_weight * reg_cpk + (1 - self.province_weight) * city_cpk
            method = 'Blend (Regional)'
        else:
            pred = city_cpk
            method = 'City Multipliers'
        
        err = ERROR_BARS.get('Low', 0.25)
        return {'predicted_cost': pred, 'cpk_low': pred*(1-err), 'cpk_high': pred*(1+err), 'method': method, 'confidence': 'Low',
                'predicted_cost': round(pred*dist, 0) if dist else round(pred, 3)}

@st.cache_resource
def load_models():
    MODEL_DIR = os.path.join(APP_DIR, 'model_export')
    try:
        with open(os.path.join(MODEL_DIR, 'config.pkl'), 'rb') as f: config = pickle.load(f)
        
        csv_path = os.path.join(MODEL_DIR, 'reference_data.csv')
        df_knn = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.read_parquet(os.path.join(MODEL_DIR, 'reference_data.parquet'))
        
        dm_path = os.path.join(MODEL_DIR, 'distance_matrix.pkl')
        distance_matrix = pickle.load(open(dm_path, 'rb')) if os.path.exists(dm_path) else {}
        
        rl_path = os.path.join(MODEL_DIR, 'rare_lane_models.pkl')
        idx_pred = IndexShrinkagePredictor(pickle.load(open(rl_path, 'rb'))) if os.path.exists(rl_path) else None
        
        bl_path = os.path.join(MODEL_DIR, 'new_lane_model_blend.pkl')
        bl_pred = BlendPredictor(pickle.load(open(bl_path, 'rb'))) if os.path.exists(bl_path) else None
        
        return {'config': config, 'df_knn': df_knn, 'distance_matrix': distance_matrix, 'index_shrink_predictor': idx_pred, 'blend_predictor': bl_pred}
    except Exception as e: return {}

models = load_models()
config = models.get('config', {})
df_knn = models.get('df_knn', pd.DataFrame())
DISTANCE_MATRIX = models.get('distance_matrix', {})
index_shrink_predictor = models.get('index_shrink_predictor')
blend_predictor = models.get('blend_predictor')

if not df_knn.empty:
    INVALID_CITY_NAMES = {'Al Farwaniyah', 'Unknown', 'unknown', '', None, 'N/A', 'n/a', 'NA', 'Murjan'}
    df_knn = df_knn[~df_knn['pickup_city'].isin(INVALID_CITY_NAMES)]
    df_knn = df_knn[~df_knn['destination_city'].isin(INVALID_CITY_NAMES)]
    VALID_CITIES_AR = set(df_knn['pickup_city'].unique()) | set(df_knn['destination_city'].unique())
else: VALID_CITIES_AR = set()

CITIES_WITHOUT_REGIONS = [] 

all_canonicals = sorted(list(set(CITY_TO_CANONICAL.values())))
pickup_cities_en = sorted(list(set([to_english_city(c) for c in all_canonicals])))
dest_cities_en = pickup_cities_en
vehicle_types_en = sorted(set(VEHICLE_TYPE_EN.values()))
commodities = sorted(set([to_english_commodity(c) for c in df_knn['commodity'].unique()]))

# ============================================
# PRICING LOGIC
# ============================================
def calculate_city_cpk_stats():
    city_stats = {}
    if len(df_knn) == 0: return {}
    outbound = df_knn.groupby('pickup_city').agg({'total_carrier_price': 'median', 'distance': 'median'}).reset_index()
    outbound['outbound_cpk'] = outbound['total_carrier_price'] / outbound['distance']
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
    stats = CITY_CPK_STATS.get(dest_city, {})
    i, o = stats.get('inbound_cpk'), stats.get('outbound_cpk')
    if i is None or o is None or o == 0: return 'Unknown', MARGIN_UNKNOWN, None
    ratio = i / o
    if ratio >= BACKHAUL_HIGH_THRESHOLD: return 'High', MARGIN_HIGH_BACKHAUL, ratio
    elif ratio >= BACKHAUL_MEDIUM_THRESHOLD: return 'Medium', MARGIN_MEDIUM_BACKHAUL, ratio
    return 'Low', MARGIN_LOW_BACKHAUL, ratio

def calculate_rental_cost(distance_km):
    if not distance_km or distance_km <= 0: return None
    days = distance_km / RENTAL_KM_PER_DAY
    return round((1.0 if days < 1 else round(days * 2) / 2) * RENTAL_COST_PER_DAY, 0)

def calculate_reference_sell(pickup_ar, dest_ar, vehicle_ar, lane_data, recommended_sell):
    recent = lane_data[lane_data['days_ago'] <= RECENCY_WINDOW]
    if len(recent) > 0: return round(recent['total_shipper_price'].median(), 0), 'Recent 90d'
    if len(lane_data) > 0: return round(lane_data['total_shipper_price'].median(), 0), 'Historical'
    if recommended_sell: return recommended_sell, 'Recommended'
    return None, 'N/A'

def get_distance(pickup_ar, dest_ar, lane_data=None, immediate_log=False, check_history=False):
    lane, rev_lane = f"{pickup_ar} → {dest_ar}", f"{dest_ar} → {pickup_ar}"
    p_can, d_can = CITY_TO_CANONICAL.get(pickup_ar, pickup_ar), CITY_TO_CANONICAL.get(dest_ar, dest_ar)
    
    if lane_data is not None and len(lane_data[lane_data['distance'] > 0]) > 0: 
        return lane_data[lane_data['distance'] > 0]['distance'].median(), 'Historical'
    
    if (pickup_ar, dest_ar) in DISTANCE_MATRIX: return DISTANCE_MATRIX[(pickup_ar, dest_ar)], 'Matrix'
    if (p_can, d_can) in DISTANCE_MATRIX: return DISTANCE_MATRIX[(p_can, d_can)], 'Matrix (canonical)'
    if (dest_ar, pickup_ar) in DISTANCE_MATRIX: return DISTANCE_MATRIX[(dest_ar, pickup_ar)], 'Matrix (reverse)'
    
    pickup_en, dest_en = to_english_city(pickup_ar), to_english_city(dest_ar)
    should_log = True
    if check_history:
        should_log = (pickup_ar in VALID_CITIES_AR or p_can in VALID_CITIES_AR) or (dest_ar in VALID_CITIES_AR or d_can in VALID_CITIES_AR)
    
    if should_log and pickup_en and dest_en:
        log_missing_distance_for_lookup(pickup_ar, dest_ar, pickup_en, dest_en, immediate=immediate_log)
    
    return 0, 'Missing'

def round_to_nearest(value, nearest):
    return int(round(value / nearest) * nearest) if value and not pd.isna(value) else None

# --- NEW: VEHICLE PRICING RULES ---
def is_core_lane(pickup_ar, dest_ar):
    CORE_CITIES = {'الرياض', 'جدة', 'الدمام'}
    p_can = CITY_TO_CANONICAL.get(pickup_ar, pickup_ar)
    d_can = CITY_TO_CANONICAL.get(dest_ar, dest_ar)
    return (p_can in CORE_CITIES) and (d_can in CORE_CITIES)

def apply_vehicle_rules(base_price, vehicle_en, is_core, weight_tons=None):
    if not base_price: return None
    v = vehicle_en.lower()
    
    if 'curtain' in v: return base_price + 100
    elif 'refrigerated' in v or 'reefer' in v: return base_price * 1.25
    elif 'lowbed' in v:
        multiplier = 2.2 if is_core else 1.85
        return base_price * multiplier
    elif 'dynna' in v:
        # Explicit or Weight based
        if 'large' in v: return base_price * 0.77
        elif 'small' in v: return base_price * 0.70
        else:
            multiplier = 0.77 if (weight_tons and weight_tons > 4.0) else 0.70
            return base_price * multiplier
    elif 'lorries' in v or 'lorry' in v: return base_price * 0.85
    elif 'closed' in v:
        if base_price <= 600: return base_price + 50
        elif base_price <= 2000: return base_price + 100
        else: return base_price + 150
    return base_price
# ----------------------------------

def calculate_prices(pickup_ar, dest_ar, requested_vehicle_ar, distance_km, lane_data=None, 
                     pickup_region_override=None, dest_region_override=None, weight=None):
    lane = f"{pickup_ar} → {dest_ar}"
    
    # 1. FORCE FLATBED DATA for the Baseline calculation
    flatbed_data = df_knn[(df_knn['lane'] == lane) & (df_knn['vehicle_type'] == DEFAULT_VEHICLE_AR)].copy()
    recent_count = len(flatbed_data[flatbed_data['days_ago'] <= RECENCY_WINDOW])
    
    if recent_count >= 1:
        base_price = flatbed_data[flatbed_data['days_ago'] <= RECENCY_WINDOW]['total_carrier_price'].median()
        model = 'Recency (Base)'
        conf = 'High' if recent_count >= 5 else ('Medium' if recent_count >= 2 else 'Low')
    elif index_shrink_predictor and index_shrink_predictor.has_lane_data(lane):
        p = index_shrink_predictor.predict(pickup_ar, dest_ar, distance_km)
        base_price, model, conf = p.get('predicted_cost'), p['method'], p['confidence']
    elif blend_predictor:
        p = blend_predictor.predict(pickup_ar, dest_ar, distance_km, pickup_region_override, dest_region_override)
        base_price, model, conf = p.get('predicted_cost'), p['method'], p['confidence']
    else:
        base_price = distance_km * 1.8 if distance_km else None
        model, conf = 'Default CPK', 'Very Low'

    # 2. APPLY VEHICLE MODIFIERS
    vehicle_en = to_english_vehicle(requested_vehicle_ar)
    is_core = is_core_lane(pickup_ar, dest_ar)
    final_buy_price = apply_vehicle_rules(base_price, vehicle_en, is_core, weight)
    
    buy_rounded = round_to_nearest(final_buy_price, 100)
    bh_prob, margin, bh_ratio = get_backhaul_probability(dest_ar)
    sell_rounded = round_to_nearest(buy_rounded * (1 + margin), 50) if buy_rounded else None
    
    req_lane_data = lane_data if lane_data is not None else df_knn[(df_knn['lane'] == lane) & (df_knn['vehicle_type'] == requested_vehicle_ar)].copy()
    ref_sell, ref_src = calculate_reference_sell(pickup_ar, dest_ar, requested_vehicle_ar, req_lane_data, sell_rounded)
    
    return {
        'buy_price': buy_rounded, 'sell_price': sell_rounded, 'ref_sell_price': ref_sell,
        'ref_sell_source': ref_src, 'rental_cost': calculate_rental_cost(distance_km),
        'target_margin': f"{margin:.0%}", 'backhaul_probability': bh_prob, 'backhaul_ratio': bh_ratio,
        'model_used': f"{model} + {vehicle_en} Rule", 'confidence': conf, 'recent_count': recent_count
    }
# ============================================
# PRICING LOGIC & HELPERS
# ============================================
@st.cache_data
def calculate_city_cpk_stats():
    city_stats = {}
    if len(df_knn) == 0: return {}
    
    # Calculate outbound/inbound medians per city
    outbound = df_knn.groupby('pickup_city').agg({'total_carrier_price': 'median', 'distance': 'median'}).reset_index()
    outbound['outbound_cpk'] = outbound['total_carrier_price'] / outbound['distance']
    
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
    Estimate backhaul probability based on Inbound/Outbound CPK ratio.
    Higher Inbound price relative to Outbound implies high demand at destination (good backhaul).
    """
    stats = CITY_CPK_STATS.get(dest_city, {})
    i_cpk = stats.get('inbound_cpk')
    o_cpk = stats.get('outbound_cpk')
    
    if i_cpk is None or o_cpk is None or o_cpk == 0:
        return 'Unknown', MARGIN_UNKNOWN, None
    
    ratio = i_cpk / o_cpk
    
    if ratio >= BACKHAUL_HIGH_THRESHOLD:
        return 'High', MARGIN_HIGH_BACKHAUL, ratio
    elif ratio >= BACKHAUL_MEDIUM_THRESHOLD:
        return 'Medium', MARGIN_MEDIUM_BACKHAUL, ratio
    else:
        return 'Low', MARGIN_LOW_BACKHAUL, ratio

def calculate_rental_cost(distance_km):
    if not distance_km or distance_km <= 0: return None
    # Estimate days needed (600km/day)
    days = distance_km / RENTAL_KM_PER_DAY
    # Round up 0.5 days (e.g. 1.2 days -> 1.5 days charge)
    days_charged = 1.0 if days < 1 else (round(days * 2) / 2)
    return round(days_charged * RENTAL_COST_PER_DAY, 0)

def calculate_reference_sell(pickup_ar, dest_ar, vehicle_ar, lane_data, recommended_sell):
    """Get a reference sell price from historical shipper prices or fall back to recommendation."""
    if lane_data is not None:
        recent = lane_data[lane_data['days_ago'] <= RECENCY_WINDOW]
        if len(recent) > 0:
            return round(recent['total_shipper_price'].median(), 0), 'Recent 90d'
        if len(lane_data) > 0:
            return round(lane_data['total_shipper_price'].median(), 0), 'Historical'
            
    if recommended_sell:
        return recommended_sell, 'Recommended'
    return None, 'N/A'

def get_distance(pickup_ar, dest_ar, lane_data=None, immediate_log=False, check_history=False):
    """
    Get distance from History -> Matrix -> Google Maps.
    """
    lane = f"{pickup_ar} → {dest_ar}"
    rev_lane = f"{dest_ar} → {pickup_ar}"
    p_can = CITY_TO_CANONICAL.get(pickup_ar, pickup_ar)
    d_can = CITY_TO_CANONICAL.get(dest_ar, dest_ar)
    
    # 1. Historical Data
    if lane_data is not None and len(lane_data[lane_data['distance'] > 0]) > 0: 
        return lane_data[lane_data['distance'] > 0]['distance'].median(), 'Historical'
    
    # 2. Distance Matrix (Exact & Canonical)
    if (pickup_ar, dest_ar) in DISTANCE_MATRIX: return DISTANCE_MATRIX[(pickup_ar, dest_ar)], 'Matrix'
    if (p_can, d_can) in DISTANCE_MATRIX: return DISTANCE_MATRIX[(p_can, d_can)], 'Matrix (canonical)'
    if (dest_ar, pickup_ar) in DISTANCE_MATRIX: return DISTANCE_MATRIX[(dest_ar, pickup_ar)], 'Matrix (reverse)'
    
    # 3. Log missing for API Lookup
    pickup_en, dest_en = to_english_city(pickup_ar), to_english_city(dest_ar)
    
    should_log = True
    if check_history:
        # Only log if at least one city is known in our historical database
        should_log = (pickup_ar in VALID_CITIES_AR or p_can in VALID_CITIES_AR) or (dest_ar in VALID_CITIES_AR or d_can in VALID_CITIES_AR)
    
    if should_log and pickup_en and dest_en:
        log_missing_distance_for_lookup(pickup_ar, dest_ar, pickup_en, dest_en, immediate=immediate_log)
    
    return 0, 'Missing'

def round_to_nearest(value, nearest):
    return int(round(value / nearest) * nearest) if value and not pd.isna(value) else None

# ============================================
# VEHICLE SPECIFIC RULES
# ============================================
def is_core_lane(pickup_ar, dest_ar):
    """Check if lane is between main hubs (Core Lane)."""
    CORE_CITIES = {'الرياض', 'جدة', 'الدمام'}
    p_can = CITY_TO_CANONICAL.get(pickup_ar, pickup_ar)
    d_can = CITY_TO_CANONICAL.get(dest_ar, dest_ar)
    return (p_can in CORE_CITIES) and (d_can in CORE_CITIES)

def apply_vehicle_rules(base_price, vehicle_en, is_core, weight_tons=None):
    """
    Apply specific pricing rules based on Flatbed baseline.
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
        # Explicit (Large/Small) or Weight based logic
        if 'large' in v:
            return base_price * 0.77
        elif 'small' in v:
            return base_price * 0.70
        else:
            # Fallback if generic 'Dynna' passed
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

def calculate_prices(pickup_ar, dest_ar, requested_vehicle_ar, distance_km, lane_data=None, 
                     pickup_region_override=None, dest_region_override=None, weight=None):
    """
    Calculate price starting with Flatbed Baseline -> Apply Vehicle Modifiers.
    """
    lane = f"{pickup_ar} → {dest_ar}"
    
    # 1. FORCE FLATBED DATA for the Baseline calculation
    flatbed_data = df_knn[(df_knn['lane'] == lane) & (df_knn['vehicle_type'] == DEFAULT_VEHICLE_AR)].copy()
    
    # Calculate BASELINE (Flatbed) Price
    recent_count = len(flatbed_data[flatbed_data['days_ago'] <= RECENCY_WINDOW])
    
    if recent_count >= 1:
        base_price = flatbed_data[flatbed_data['days_ago'] <= RECENCY_WINDOW]['total_carrier_price'].median()
        model = 'Recency (Base)'
        conf = 'High' if recent_count >= 5 else ('Medium' if recent_count >= 2 else 'Low')
    elif index_shrink_predictor and index_shrink_predictor.has_lane_data(lane):
        p = index_shrink_predictor.predict(pickup_ar, dest_ar, distance_km)
        base_price, model, conf = p.get('predicted_cost'), p['method'], p['confidence']
    elif blend_predictor:
        p = blend_predictor.predict(pickup_ar, dest_ar, distance_km,
                                    pickup_region_override=pickup_region_override,
                                    dest_region_override=dest_region_override)
        base_price, model, conf = p.get('predicted_cost'), p['method'], p['confidence']
    else:
        base_price = distance_km * 1.8 if distance_km else None
        model, conf = 'Default CPK', 'Very Low'

    # 2. APPLY VEHICLE MODIFIERS
    vehicle_en = to_english_vehicle(requested_vehicle_ar)
    is_core = is_core_lane(pickup_ar, dest_ar)
    
    final_buy_price = apply_vehicle_rules(base_price, vehicle_en, is_core, weight)
    
    # Rounding
    buy_rounded = round_to_nearest(final_buy_price, 100)
    
    # Calculate sell price (Margin logic)
    bh_prob, margin, bh_ratio = get_backhaul_probability(dest_ar)
    sell_rounded = round_to_nearest(buy_rounded * (1 + margin), 50) if buy_rounded else None
    
    # Get reference sell (Visual only - using original requested vehicle data if available)
    req_lane_data = lane_data if lane_data is not None else df_knn[(df_knn['lane'] == lane) & (df_knn['vehicle_type'] == requested_vehicle_ar)].copy()
    ref_sell, ref_src = calculate_reference_sell(pickup_ar, dest_ar, requested_vehicle_ar, req_lane_data, sell_rounded)
    
    return {
        'buy_price': buy_rounded, 
        'sell_price': sell_rounded, 
        'base_flatbed_price': base_price,
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
        
    dist, dist_src = get_distance(pickup_ar, dest_ar, lane_data, immediate_log=True)
    if dist == 0: dist, dist_src = 500, 'Default'
    
    # Calculate Price
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
    """
    if not vehicle_ar or vehicle_ar in ['', 'Auto', 'auto']: vehicle_ar = DEFAULT_VEHICLE_AR
    lane_data = df_knn[(df_knn['lane'] == f"{pickup_ar} → {dest_ar}") & (df_knn['vehicle_type'] == vehicle_ar)].copy()
    
    # Use override distance if provided, otherwise look it up
    dist_source = 'User Provided'
    if dist_override is not None and dist_override > 0:
        dist = dist_override
    else:
        # Use immediate_log=False for bulk pricing
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
if blend_predictor: model_status.append(f"✅ Blend ({'Province' if blend_predictor.using_provinces else 'Region'})")
dist_status = f"✅ {len(DISTANCE_MATRIX):,} distances" if DISTANCE_MATRIX else ""
st.caption(f"ML-powered pricing | Domestic | {' | '.join(model_status)} | {dist_status}")

if CITIES_WITHOUT_REGIONS:
    with st.expander(f"⚠️ {len(CITIES_WITHOUT_REGIONS)} cities in historical data missing from canonicals CSV", expanded=False):
        st.warning("**Action Required:** These cities exist in historical trip data but are NOT in city_normalization.csv.")
        for city in CITIES_WITHOUT_REGIONS: st.code(city)
        st.info("Add these to city_normalization.csv with columns: variant, canonical, english, region")

tab1, tab2 = st.tabs(["🎯 Single Route Pricing", "📦 Bulk Route Lookup"])

# --- TAB 1: SINGLE ROUTE PRICING ---
with tab1:
    st.subheader("📋 Route Information")
    if not pickup_cities_en: st.error("City list is empty. Check normalization file."); st.stop()
    
    if 'single_pickup' not in st.session_state: st.session_state.single_pickup = pickup_cities_en[0] if pickup_cities_en else ''
    if 'single_dest' not in st.session_state: st.session_state.single_dest = dest_cities_en[1] if len(dest_cities_en)>1 else ''
    
    c1, c2, c3 = st.columns(3)
    pickup_en = c1.selectbox("Pickup", pickup_cities_en, key='single_pickup')
    dest_en = c2.selectbox("Destination", dest_cities_en, key='single_dest')
    vehicle_en = c3.selectbox("Vehicle", vehicle_types_en)
    
    # Optional Inputs
    c1, c2, c3 = st.columns(3)
    c1.selectbox("Container", ['Auto-detect', 'Yes', 'No'], key='single_container')
    comm_sel = c2.selectbox("Commodity", ['Auto-detect'] + commodities, key='single_commodity')
    w = c3.number_input("Weight (Tons)", 0.0, 100.0, 0.0, step=1.0, help="0 = Auto")
    
    if st.button("🎯 Generate Pricing", type="primary", use_container_width=True):
        comm_in = None if comm_sel == 'Auto-detect' else to_arabic_commodity(comm_sel)
        weight_in = None if w == 0 else w
        
        result = price_single_route(to_arabic_city(pickup_en), to_arabic_city(dest_en), to_arabic_vehicle(vehicle_en), comm_in, weight_in)
        
        st.info(f"**{pickup_en} → {dest_en}** | 🚛 {result['Vehicle_Type']} | 📏 {result['Distance_km']:.0f} km")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🛒 BUY PRICE", f"{result['Buy_Price']:,} SAR" if result['Buy_Price'] else "N/A")
        c2.metric("💵 SELL PRICE", f"{result['Rec_Sell_Price']:,} SAR" if result.get('Rec_Sell_Price') else "N/A")
        c3.metric("Margin", result['Target_Margin'])
        c4.metric("Model", result['Model_Used'])
        
        if result['Is_Rare_Lane']: st.warning("⚠️ **Rare Lane** - Using model prediction.")

# --- TAB 2: BULK ROUTE LOOKUP (WIZARD) ---
with tab2:
    st.subheader("📦 Bulk Route Lookup")
    
    # State Management for Multi-Stage Process
    if 'bulk_stage' not in st.session_state: st.session_state.bulk_stage = 'upload'
    if 'runtime_variants' not in st.session_state: st.session_state.runtime_variants = {}
    
    upl = st.file_uploader("Upload CSV", type=['csv'])
    
    if upl:
        # Check if new file uploaded (reset state)
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != upl.name:
            st.session_state.bulk_stage = 'upload'
            st.session_state.last_uploaded_file = upl.name
            st.session_state.bulk_df = pd.read_csv(upl)
            # Clear previous data
            for k in ['bulk_results', 'fuzzy_suggestions', 'coord_df', 'unmatched_list']:
                if k in st.session_state: del st.session_state[k]
            
        r_df = st.session_state.bulk_df
        st.success(f"✅ Loaded {len(r_df)} routes")
        
        # Vehicle Checkboxes
        st.markdown("##### 🔧 Settings")
        st.caption("Select vehicle types to apply to ALL routes.")
        v_cols = st.columns(4)
        selected_vehicles = []
        if 'bulk_selected_vehicles' not in st.session_state: st.session_state.bulk_selected_vehicles = ['Flatbed Trailer']
            
        for i, v in enumerate(vehicle_types_en):
            with v_cols[i%4]:
                is_selected = st.checkbox(v, value=(v in st.session_state.bulk_selected_vehicles), key=f"chk_{v}")
                if is_selected: selected_vehicles.append(v)
        
        st.session_state.bulk_selected_vehicles = selected_vehicles
        
        if st.button("🔍 Start Processing", type="primary"):
            st.session_state.bulk_stage = 'scanning'
            st.rerun()
            
        # STAGE 1: SCAN FOR UNMATCHED CITIES
        if st.session_state.bulk_stage == 'scanning':
            with st.spinner("Scanning for new cities..."):
                unmatched = set()
                for i in range(len(r_df)):
                    row = r_df.iloc[i]
                    p_raw = str(row.iloc[0]).strip() if len(row) > 0 else ''
                    d_raw = str(row.iloc[1]).strip() if len(row) > 1 else ''
                    
                    _, p_ok = normalize_city(p_raw)
                    _, d_ok = normalize_city(d_raw)
                    
                    if not p_ok and p_raw: unmatched.add(p_raw)
                    if not d_ok and d_raw: unmatched.add(d_raw)
                
                if unmatched:
                    st.session_state.unmatched_list = sorted(list(unmatched))
                    st.session_state.bulk_stage = 'fuzzy_review'
                else:
                    st.session_state.bulk_stage = 'processing'
                st.rerun()

        # STAGE 2: FUZZY MATCH REVIEW
        if st.session_state.bulk_stage == 'fuzzy_review':
            st.warning(f"⚠️ Found {len(st.session_state.unmatched_list)} unmatched cities")
            st.info("Please review suggested matches. Check 'Approve' to map them permanently.")
            
            if 'fuzzy_suggestions' not in st.session_state:
                suggestions = []
                for city in st.session_state.unmatched_list:
                    match, score = find_best_match(city, all_canonicals)
                    suggestions.append({
                        'Original': city,
                        'Suggested Match': match if match else 'No match',
                        'Confidence': f"{score:.0%}",
                        'Approve?': False,
                        'Raw_Score': score,
                        'Raw_Match': match
                    })
                st.session_state.fuzzy_suggestions = pd.DataFrame(suggestions)
            
            edited_df = st.data_editor(
                st.session_state.fuzzy_suggestions,
                column_config={
                    "Approve?": st.column_config.CheckboxColumn("Approve Match?", default=False),
                    "Raw_Score": None, "Raw_Match": None
                },
                disabled=["Original", "Suggested Match", "Confidence"],
                hide_index=True,
                use_container_width=True
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirm Matches & Continue", type="primary"):
                    approved = edited_df[edited_df['Approve?'] == True]
                    new_variants = []
                    
                    for _, row in approved.iterrows():
                        if row['Raw_Match']:
                            st.session_state.runtime_variants[row['Original']] = row['Raw_Match']
                            new_variants.append({'variant': row['Original'], 'canonical': row['Raw_Match'], 'score': row['Raw_Score']})
                    
                    if new_variants:
                        log_variants_to_sheet(new_variants)
                        st.toast(f"Saved {len(new_variants)} new city mappings!")
                    
                    remaining = []
                    for city in st.session_state.unmatched_list:
                        if city not in st.session_state.runtime_variants: remaining.append(city)
                    
                    if remaining:
                        st.session_state.remaining_unmatched = remaining
                        st.session_state.bulk_stage = 'coord_input'
                    else:
                        st.session_state.bulk_stage = 'processing'
                    st.rerun()
            
            with col2:
                if st.button("Skip Fuzzy Matching"):
                    st.session_state.remaining_unmatched = st.session_state.unmatched_list
                    st.session_state.bulk_stage = 'coord_input'
                    st.rerun()

        # STAGE 3: COORDINATE INPUT
        if st.session_state.bulk_stage == 'coord_input':
            st.info("🌍 Location Required for New Cities")
            st.write("For the remaining unmatched cities, please provide Latitude/Longitude so we can determine the region.")
            
            if 'coord_df' not in st.session_state:
                st.session_state.coord_df = pd.DataFrame([{'City': c, 'Lat': 0.0, 'Lon': 0.0} for c in st.session_state.remaining_unmatched])
                
            coords_edited = st.data_editor(
                st.session_state.coord_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Lat": st.column_config.NumberColumn("Latitude", format="%.5f"),
                    "Lon": st.column_config.NumberColumn("Longitude", format="%.5f")
                }
            )
            
            if st.button("🚀 Calculate Prices", type="primary"):
                if 'temp_coords' not in st.session_state: st.session_state.temp_coords = {}
                for _, row in coords_edited.iterrows():
                    if row['Lat'] != 0.0 or row['Lon'] != 0.0:
                        st.session_state.temp_coords[row['City']] = (row['Lat'], row['Lon'])
                st.session_state.bulk_stage = 'processing'
                st.rerun()

        # STAGE 4: PROCESSING
        if st.session_state.bulk_stage == 'processing':
            res, missing_dists = [], []
            prog = st.progress(0)
            status = st.empty()
            
            vehicles_to_run = [to_arabic_vehicle(v) for v in st.session_state.bulk_selected_vehicles]
            if not vehicles_to_run: vehicles_to_run = [DEFAULT_VEHICLE_AR]
            
            for i in range(len(r_df)):
                row = r_df.iloc[i]
                p_raw = str(row.iloc[0]).strip() if len(row) > 0 else ''
                d_raw = str(row.iloc[1]).strip() if len(row) > 1 else ''
                
                # Check for explicit Weight column (Col 9 / Index 8)
                weight_val = None
                if len(row) > 8:
                    try: 
                        val = row.iloc[8]
                        if pd.notna(val): weight_val = float(str(val).strip())
                    except: pass

                # Check for explicit Distance
                dist_override = None
                if len(row) > 2:
                    try:
                        val = row.iloc[2]
                        if pd.notna(val) and str(val).strip() not in ['', 'nan']:
                            dist_override = float(str(val).replace(',', '').strip())
                    except: pass

                p_ar, _ = normalize_city(p_raw)
                d_ar, _ = normalize_city(d_raw)
                if not p_ar: p_ar = p_raw
                if not d_ar: d_ar = d_raw
                
                p_reg_over, d_reg_over = None, None
                if 'temp_coords' in st.session_state:
                    if p_raw in st.session_state.temp_coords:
                        lat, lon = st.session_state.temp_coords[p_raw]
                        _, p_reg_over = get_province_from_coordinates(lat, lon)
                    if d_raw in st.session_state.temp_coords:
                        lat, lon = st.session_state.temp_coords[d_raw]
                        _, d_reg_over = get_province_from_coordinates(lat, lon)

                for v_ar in vehicles_to_run:
                    route_res = lookup_route_stats(p_ar, d_ar, v_ar, 
                                                   dist_override=dist_override, 
                                                   pickup_region_override=p_reg_over, 
                                                   dest_region_override=d_reg_over,
                                                   weight=weight_val)
                    route_res['Original_From'] = p_raw
                    route_res['Original_To'] = d_raw
                    
                    if route_res['Distance'] == 0:
                        missing_dists.append({'From': route_res['From'], 'To': route_res['To'], 'Vehicle': route_res['Vehicle'], 'Status': 'Missing'})
                    res.append(route_res)
                
                prog.progress((i + 1) / len(r_df))
                status.text(f"Processing {i+1}/{len(r_df)}")
            
            st.session_state.bulk_results = pd.DataFrame(res)
            st.session_state.bulk_missing_distances = pd.DataFrame(missing_dists).drop_duplicates(subset=['From', 'To'])
            flush_matched_distances_to_sheet()
            st.session_state.bulk_stage = 'done'
            st.rerun()

        # STAGE 5: RESULTS & SAVE
        if st.session_state.bulk_stage == 'done':
            if 'bulk_missing_distances' in st.session_state and not st.session_state.bulk_missing_distances.empty:
                st.error(f"📏 {len(st.session_state.bulk_missing_distances)} Distances Missing")
                st.caption("We've logged these to the Google Sheet. Please update them in the Admin section.")
                with st.expander("View Missing Distances"): st.dataframe(st.session_state.bulk_missing_distances, use_container_width=True)
            
            if 'bulk_results' in st.session_state:
                res_df = st.session_state.bulk_results
                st.subheader("📊 Final Results")
                st.dataframe(res_df, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Save 🚀")
                
                rfq_sheet_url = st.secrets.get('RFQ_url', '')
                if rfq_sheet_url:
                    st.markdown(f"Target Spreadsheet: [**RFQ Pricing Sheet**]({rfq_sheet_url})")
                    default_sheet_name = f"Pricing_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    sheet_name = st.text_input("Sheet Name", value=default_sheet_name, key="rfq_sheet_name_wiz")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("☁️ Upload to Google Sheet", type="secondary", use_container_width=True):
                            if not sheet_name: st.error("Enter sheet name")
                            else:
                                ok, msg = upload_to_rfq_sheet(res_df, sheet_name)
                                if ok: st.success(msg); st.link_button("🔗 Open Sheet", rfq_sheet_url)
                                else: st.error(msg)
                    with c2: st.download_button("📥 Download CSV", res_df.to_csv(index=False), "results.csv", "text/csv", use_container_width=True)
            
            if st.button("🔄 Start New Upload"):
                del st.session_state.bulk_stage
                st.rerun()

    # ============================================
    # DISTANCE ADMIN (CENTERED UI)
    # ============================================
    st.markdown("---")
    with st.expander("📏 Admin: Update Distances from Google Sheets"):
        # Inject CSS to strictly force centering on Metric elements
        st.markdown("""
            <style>
            div[data-testid="stMetric"] { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; width: 100%; }
            div[data-testid="stMetricLabel"] { justify-content: center; width: 100%; margin: auto; }
            div[data-testid="stMetricValue"] { justify-content: center; width: 100%; margin: auto; }
            </style>
            """, unsafe_allow_html=True)

        resolved = get_resolved_distances_from_sheet()
        failed = get_failed_distances_from_sheet()
        
        st.markdown("<h3 style='text-align: center;'>📊 Status Dashboard</h3>", unsafe_allow_html=True)
        
        _, m1, m2, m3, _ = st.columns([1, 2, 2, 2, 1])
        m1.metric("Ready to Import", len(resolved))
        m2.metric("Failed Lookups", len(failed))
        m3.metric("Current Pickle Size", f"{len(DISTANCE_MATRIX):,}")
        
        if resolved or failed:
            st.markdown("<h4 style='text-align: left;'>📋 Details</h4>", unsafe_allow_html=True)
            t1, t2 = st.tabs(["✅ Ready to Import", "⚠️ Failed Lookups"])
            with t1:
                if resolved:
                    st.markdown("<p style='text-align: left; color: gray;'>These distances will be added to your local database.</p>", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(resolved), use_container_width=True, hide_index=True)
                else: st.info("No pending distances.")
            with t2:
                if failed:
                    st.markdown("<p style='text-align: left; color: orange;'>These routes failed API lookup. Manually enter distances in Column G.</p>", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(failed), use_container_width=True, hide_index=True)
                else: st.success("No failed lookups!")

        st.markdown("---")
        st.markdown("<h5 style='text-align: center;'>💾 Save & Sync</h5>", unsafe_allow_html=True)
        _, c1, c2, c3, _ = st.columns([1, 2, 2, 2, 1])
        
        with c1: 
            if st.button("🔍 Refresh Sheet Data", use_container_width=True): st.rerun()
        with c2: 
            if st.button("🔄 Import to Pickle", type="primary", disabled=len(resolved)==0, use_container_width=True):
                success, count, message = update_distance_pickle_from_sheet()
                if success: st.success(message); st.balloons(); st.cache_resource.clear()
                else: st.error(message)
        with c3:
            pkl_path = os.path.join(APP_DIR, 'model_export', 'distance_matrix.pkl')
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f: pkl_data = f.read()
                st.download_button("📥 Download .pkl", pkl_data, "distance_matrix.pkl", "application/octet-stream", use_container_width=True, type="secondary")
