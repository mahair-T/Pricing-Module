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

def flush_matched_distances_to_sheet():
    """
    Flush pending matched distances to Google Sheet.
    Call at the end of bulk operations.
    
    Returns: (success, count) tuple
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
    'Lowbed Trailer': 'Lowbed Trailer',
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

def calculate_prices(pickup_ar, dest_ar, vehicle_ar, distance_km, lane_data=None,
                     pickup_region_override=None, dest_region_override=None):
    """
    Calculate buy/sell prices using the pricing cascade.
    
    PRICING CASCADE (in order of preference):
    1. RECENCY: If recent loads exist (within RECENCY_WINDOW days), use median
    2. INDEX+SHRINKAGE: If ANY historical data exists for this lane, use Index+Shrinkage model
    3. PROVINCE BLEND: For NEW lanes (no history), use 70% Province CPK + 30% City decomposition
    4. REGIONAL BLEND: If Province Blend fails (missing province data), use 5-region CPK instead
    5. CITY MULTIPLIERS: If no geographic data at all, use city multipliers only
    6. DEFAULT: Last resort - use default CPK × distance
    
    Args:
        pickup_ar: Pickup city in Arabic
        dest_ar: Destination city in Arabic
        vehicle_ar: Vehicle type in Arabic
        distance_km: Distance in km
        lane_data: Optional pre-filtered lane data
        pickup_region_override: Optional region override from coordinates
        dest_region_override: Optional region override from coordinates
    
    Returns dict with buy_price, sell_price, model_used, confidence, etc.
    """
    lane = f"{pickup_ar} → {dest_ar}"
    if lane_data is None: lane_data = df_knn[(df_knn['lane'] == lane) & (df_knn['vehicle_type'] == vehicle_ar)].copy()
    
    recent_count = len(lane_data[lane_data['days_ago'] <= RECENCY_WINDOW])
    
    # PRICING CASCADE
    if recent_count >= 1:
        # 1. RECENCY MODEL: Use median of recent loads
        buy_price = lane_data[lane_data['days_ago'] <= RECENCY_WINDOW]['total_carrier_price'].median()
        model = 'Recency'
        conf = 'High' if recent_count >= 5 else ('Medium' if recent_count >= 2 else 'Low')
    elif index_shrink_predictor and index_shrink_predictor.has_lane_data(lane):
        # 2. INDEX + SHRINKAGE: Lane has ANY historical data (even if not recent)
        p = index_shrink_predictor.predict(pickup_ar, dest_ar, distance_km)
        buy_price, model, conf = p.get('predicted_cost'), p['method'], p['confidence']
    elif blend_predictor:
        # 3-5. BLEND MODEL: NEW lane with NO historical data
        # Internally tries: Province CPK → Regional CPK → City Multipliers
        # Pass region overrides from coordinates if available
        p = blend_predictor.predict(pickup_ar, dest_ar, distance_km,
                                    pickup_region_override=pickup_region_override,
                                    dest_region_override=dest_region_override)
        buy_price, model, conf = p.get('predicted_cost'), p['method'], p['confidence']
    else:
        # 6. DEFAULT: Last resort fallback (no models available)
        buy_price = distance_km * 1.8 if distance_km else None
        model, conf = 'Default CPK', 'Very Low'
    
    # Round buy price to nearest 100 SAR
    buy_rounded = round_to_nearest(buy_price, 100)
    
    # Calculate sell price based on backhaul probability
    bh_prob, margin, bh_ratio = get_backhaul_probability(dest_ar)
    sell_rounded = round_to_nearest(buy_rounded * (1 + margin), 50) if buy_rounded else None
    
    # Get reference sell price from historical shipper prices
    ref_sell, ref_src = calculate_reference_sell(pickup_ar, dest_ar, vehicle_ar, lane_data, sell_rounded)
    
    return {
        'buy_price': buy_rounded, 'sell_price': sell_rounded, 'ref_sell_price': round_to_nearest(ref_sell, 50),
        'ref_sell_source': ref_src, 'rental_cost': calculate_rental_cost(distance_km),
        'target_margin': f"{margin:.0%}", 'backhaul_probability': bh_prob, 'backhaul_ratio': round(bh_ratio, 2) if bh_ratio else None,
        'model_used': model, 'confidence': conf, 'recent_count': recent_count
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
    
    pricing = calculate_prices(pickup_ar, dest_ar, vehicle_ar, dist, lane_data)
    
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
                       pickup_region_override=None, dest_region_override=None):
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
    """
    if not vehicle_ar or vehicle_ar in ['', 'Auto', 'auto']: vehicle_ar = DEFAULT_VEHICLE_AR
    lane_data = df_knn[(df_knn['lane'] == f"{pickup_ar} → {dest_ar}") & (df_knn['vehicle_type'] == vehicle_ar)].copy()
    
    # Use override distance if provided, otherwise look it up
    dist_source = 'User Provided'
    if dist_override is not None and dist_override > 0:
        dist = dist_override
    else:
        # Use immediate_log=False for bulk pricing (queues errors for batch write)
        dist, dist_source = get_distance(pickup_ar, dest_ar, lane_data, immediate_log=False, check_history=check_history)
    
    pricing = calculate_prices(pickup_ar, dest_ar, vehicle_ar, dist, lane_data,
                               pickup_region_override=pickup_region_override,
                               dest_region_override=dest_region_override)
    return {
        'From': to_english_city(pickup_ar), 'To': to_english_city(dest_ar), 'Distance': int(dist) if dist else 0,
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

tab1, tab2 = st.tabs(["🎯 Single Route Pricing", "📦 Bulk Route Lookup"])

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
    
    st.markdown(f"""
    **Upload a CSV to get pricing for each route.**
    
    **Columns (by position, names ignored):**
    1. **Pickup City** (required)
    2. **Destination City** (required)
    3. **Distance** (optional - used temporarily, NOT saved to references)
    4. **Vehicle Type** (optional - default: Flatbed Trailer)
    5. **Pickup Latitude** (optional - for region detection if city unmatched)
    6. **Pickup Longitude** (optional - for region detection if city unmatched)
    7. **Dropoff Latitude** (optional - for region detection if city unmatched)
    8. **Dropoff Longitude** (optional - for region detection if city unmatched)
    
    ⚠️ **Note:** User-provided distances and coordinates are used temporarily for pricing but are **NOT** saved to our reference data. They are logged to "Bulk Adjustments" sheet for review.
    
    **Output columns:**
    - Distance, Buy Price (rounded to 100), Sell Price (rounded to 50)
    - Target Margin, Backhaul Probability
    - Model Used, Confidence, Recent Count
    """)
    
    # Template button - populates Template sheet in RFQ spreadsheet
    if st.button("📋 Open Template in Google Sheets", help="Opens/updates the Template sheet in the RFQ spreadsheet"):
        template_df = pd.DataFrame({
            'Pickup': ['Jeddah', 'Riyadh', 'Unknown City'], 
            'Destination': ['Riyadh', 'Dammam', 'Jeddah'], 
            'Distance': ['', '', '450'],
            'Vehicle_Type': ['Flatbed Trailer', '', ''],
            'Pickup_Lat': ['', '', '24.7136'],
            'Pickup_Lon': ['', '', '46.6753'],
            'Dropoff_Lat': ['', '', '21.4858'],
            'Dropoff_Lon': ['', '', '39.1925']
        })
        
        success, msg = populate_rfq_template_sheet(template_df)
        if success:
            rfq_url = st.secrets.get('RFQ_url', '')
            st.success(msg)
            if rfq_url:
                # Create a direct link to the Template sheet
                base_url = rfq_url.split('/edit')[0]
                st.markdown(f"[🔗 Open Template Sheet]({base_url}/edit#gid=0)")
        else:
            st.error(msg)
    
    upl = st.file_uploader("Upload CSV", type=['csv'])
    if upl:
        try:
            r_df = pd.read_csv(upl)
            st.success(f"✅ Loaded {len(r_df)} routes")
            
            with st.expander("📋 Preview"):
                st.dataframe(r_df.head(10), use_container_width=True)
            
            if st.button("🔍 Look Up All Routes", type="primary", use_container_width=True):
                res = []
                unmat_cities = []  # Unmatched cities
                missing_distances = []  # Missing distances
                prog = st.progress(0)
                status_text = st.empty()
                
                # Get column values by position (iloc), not by name
                for i in range(len(r_df)):
                    row = r_df.iloc[i]
                    
                    # Column 1: Pickup (required)
                    p_raw = str(row.iloc[0]).strip() if len(row) > 0 else ''
                    
                    # Column 2: Destination (required)
                    d_raw = str(row.iloc[1]).strip() if len(row) > 1 else ''
                    
                    # Column 3: Distance (optional - used temporarily only)
                    dist_override = None
                    user_provided_distance = False
                    if len(row) > 2:
                        dist_val = row.iloc[2]
                        if pd.notna(dist_val) and str(dist_val).strip() not in ['', 'nan', 'None']:
                            try:
                                dist_override = float(str(dist_val).replace(',', '').strip())
                                if dist_override > 0:
                                    user_provided_distance = True
                            except ValueError:
                                pass
                    
                    # Column 4: Vehicle Type (optional)
                    v_raw = ''
                    if len(row) > 3:
                        v_raw = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ''
                    
                    # Columns 5-8: Lat/Lon coordinates (optional)
                    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = None, None, None, None
                    user_provided_coords = False
                    
                    if len(row) > 4:
                        try:
                            val = row.iloc[4]
                            if pd.notna(val) and str(val).strip() not in ['', 'nan', 'None']:
                                pickup_lat = float(str(val).strip())
                        except (ValueError, TypeError):
                            pass
                    
                    if len(row) > 5:
                        try:
                            val = row.iloc[5]
                            if pd.notna(val) and str(val).strip() not in ['', 'nan', 'None']:
                                pickup_lon = float(str(val).strip())
                        except (ValueError, TypeError):
                            pass
                    
                    if len(row) > 6:
                        try:
                            val = row.iloc[6]
                            if pd.notna(val) and str(val).strip() not in ['', 'nan', 'None']:
                                dropoff_lat = float(str(val).strip())
                        except (ValueError, TypeError):
                            pass
                    
                    if len(row) > 7:
                        try:
                            val = row.iloc[7]
                            if pd.notna(val) and str(val).strip() not in ['', 'nan', 'None']:
                                dropoff_lon = float(str(val).strip())
                        except (ValueError, TypeError):
                            pass
                    
                    # Check if user provided any coordinates
                    if pickup_lat is not None or pickup_lon is not None or dropoff_lat is not None or dropoff_lon is not None:
                        user_provided_coords = True
                    
                    # Detect provinces from coordinates if provided
                    pickup_province, pickup_region = None, None
                    dropoff_province, dropoff_region = None, None
                    
                    if pickup_lat is not None and pickup_lon is not None:
                        pickup_province, pickup_region = get_province_from_coordinates(pickup_lat, pickup_lon)
                    
                    if dropoff_lat is not None and dropoff_lon is not None:
                        dropoff_province, dropoff_region = get_province_from_coordinates(dropoff_lat, dropoff_lon)
                    
                    # Normalize cities
                    p_ar, p_ok = normalize_city(p_raw)
                    d_ar, d_ok = normalize_city(d_raw)
                    
                    if not p_ok: 
                        classification = classify_unmatched_city(p_raw)
                        unmat_cities.append({
                            'Row': i+1, 
                            'Col': 'Pickup (Col 1)', 
                            'Val': p_raw,
                            'Error': classification['error_type'],
                            'Action': classification['action'],
                            'Detected_Province': pickup_province or '',
                            'Detected_Region': pickup_region or ''
                        })
                        log_exception(classification['error_type'], {
                            'row': i+1, 
                            'column': 'Pickup', 
                            'original_value': p_raw,
                            'message': classification['message'],
                            'action': classification['action'],
                            'in_historical_data': classification['in_historical_data'],
                            'detected_province': pickup_province,
                            'detected_region': pickup_region
                        })
                    
                    if not d_ok: 
                        classification = classify_unmatched_city(d_raw)
                        unmat_cities.append({
                            'Row': i+1, 
                            'Col': 'Destination (Col 2)', 
                            'Val': d_raw,
                            'Error': classification['error_type'],
                            'Action': classification['action'],
                            'Detected_Province': dropoff_province or '',
                            'Detected_Region': dropoff_region or ''
                        })
                        log_exception(classification['error_type'], {
                            'row': i+1, 
                            'column': 'Destination', 
                            'original_value': d_raw,
                            'message': classification['message'],
                            'action': classification['action'],
                            'in_historical_data': classification['in_historical_data'],
                            'detected_province': dropoff_province,
                            'detected_region': dropoff_region
                        })
                    
                    v_ar = to_arabic_vehicle(v_raw) if v_raw not in ['', 'nan', 'None', 'Auto'] else DEFAULT_VEHICLE_AR
                    
                    # Log to Bulk Adjustments sheet if user provided distance or coordinates
                    if user_provided_distance or user_provided_coords:
                        log_bulk_adjustment(
                            row_num=i+1,
                            pickup_city=p_raw,
                            dest_city=d_raw,
                            pickup_en=to_english_city(p_ar) if p_ok else p_raw,
                            dest_en=to_english_city(d_ar) if d_ok else d_raw,
                            user_distance=dist_override if user_provided_distance else None,
                            pickup_lat=pickup_lat,
                            pickup_lon=pickup_lon,
                            dropoff_lat=dropoff_lat,
                            dropoff_lon=dropoff_lon,
                            pickup_province=pickup_province,
                            dropoff_province=dropoff_province,
                            pickup_region=pickup_region,
                            dropoff_region=dropoff_region
                        )
                    
                    # Pass distance override and regions to lookup function
                    route_result = lookup_route_stats(
                        p_ar, d_ar, v_ar, 
                        dist_override=dist_override,
                        pickup_region_override=pickup_region,
                        dest_region_override=dropoff_region
                    )
                    
                    # Track missing distances
                    if route_result.get('Distance', 0) == 0 or route_result.get('Distance_Source') == 'Missing':
                        missing_distances.append({
                            'Row': i+1,
                            'From': route_result.get('From', p_raw),
                            'To': route_result.get('To', d_raw),
                            'Distance': route_result.get('Distance', 0),
                            'Status': 'Missing - needs Google Maps lookup'
                        })
                    
                    res.append(route_result)
                    prog.progress((i+1)/len(r_df))
                    status_text.text(f"Looking up {i+1}/{len(r_df)}: {p_raw} → {d_raw}")
                
                status_text.text("✅ Complete!")
                st.session_state.bulk_results = pd.DataFrame(res)
                st.session_state.bulk_unmatched_cities = pd.DataFrame(unmat_cities)
                st.session_state.bulk_missing_distances = pd.DataFrame(missing_distances)
                
                # Flush any pending error logs to Google Sheets in one batch
                flushed_ok, flushed_count = flush_error_log_to_sheet()
                if flushed_count > 0:
                    st.caption(f"📝 Logged {flushed_count} exceptions to error sheet")
                
                # Flush any pending matched distances to Google Sheets
                dist_ok, dist_count = flush_matched_distances_to_sheet()
                if dist_count > 0:
                    st.caption(f"📏 Logged {dist_count} missing distances for Google Maps lookup")
                
                # Flush any pending bulk adjustments to Google Sheets
                adj_ok, adj_count = flush_bulk_adjustments_to_sheet()
                if adj_count > 0:
                    st.caption(f"📋 Logged {adj_count} user-provided adjustments for review")
                    
        except Exception as e: st.error(f"Error: {e}")

    # Results Display (Outside indentation so it persists)
    if 'bulk_results' in st.session_state and not st.session_state.bulk_results.empty:
        res_df = st.session_state.bulk_results
        st.markdown("---")
        st.subheader("📊 Results")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

    # Separate tables for unmatched cities and missing distances
    st.markdown("---")
    
    # Table 1: Unmatched Cities
    if 'bulk_unmatched_cities' in st.session_state and not st.session_state.bulk_unmatched_cities.empty:
        st.subheader("⚠️ Unmatched Cities")
        st.warning(f"{len(st.session_state.bulk_unmatched_cities)} city names could not be matched to our database")
        st.dataframe(st.session_state.bulk_unmatched_cities, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Download Unmatched Cities", 
            st.session_state.bulk_unmatched_cities.to_csv(index=False), 
            "unmatched_cities.csv", 
            "text/csv",
            key="download_unmatched_cities"
        )
    
    # Table 2: Missing Distances
    if 'bulk_missing_distances' in st.session_state and not st.session_state.bulk_missing_distances.empty:
        st.subheader("📏 Missing Distances")
        st.info(f"{len(st.session_state.bulk_missing_distances)} routes have missing distances (logged for Google Maps lookup)")
        st.dataframe(st.session_state.bulk_missing_distances, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Download Missing Distances", 
            st.session_state.bulk_missing_distances.to_csv(index=False), 
            "missing_distances.csv", 
            "text/csv",
            key="download_missing_distances"
        )

    # Download & Cloud Upload Section (after unmatched sections)
    if 'bulk_results' in st.session_state and not st.session_state.bulk_results.empty:
        res_df = st.session_state.bulk_results
        st.markdown("---")
        st.subheader("Save 🚀")
        
        rfq_sheet_url = st.secrets.get('RFQ_url', '')
        
        if rfq_sheet_url:
            # Hyperlinked text for the sheet
            st.markdown(f"Target Spreadsheet: [**RFQ Pricing Sheet**]({rfq_sheet_url})")
            
            # Prompt for sheet name
            default_sheet_name = f"Pricing_{datetime.now().strftime('%Y%m%d_%H%M')}"
            sheet_name = st.text_input(
                "Sheet Name", 
                value=default_sheet_name,
                help="Enter a unique name for the new sheet tab. This prevents overwriting existing data.",
                key="rfq_sheet_name"
            )
            
            # Create two columns for the buttons to sit side-by-side
            col1, col2 = st.columns(2)
            
            with col1:
                # Upload Button (Left)
                upload_clicked = st.button("☁️ Upload to Google Sheet", type="secondary", key="upload_to_rfq", use_container_width=True)
                
            with col2:
                # Download Button (Right) - Changed type to 'secondary' to match color and removed 'primary' (red)
                st.download_button(
                    "📥 Download Results CSV", 
                    res_df.to_csv(index=False), 
                    "pricing_results.csv", 
                    "text/csv", 
                    type="secondary",
                    key="download_results_csv",
                    use_container_width=True
                )

            # Upload Logic
            if upload_clicked:
                if not sheet_name or sheet_name.strip() == '':
                    st.error("Please enter a sheet name")
                else:
                    upload_progress = st.progress(0)
                    upload_status = st.empty()
                    
                    def update_progress(rows_done, total_rows, batch_num):
                        upload_progress.progress(rows_done / total_rows)
                        upload_status.text(f"Uploaded {rows_done:,}/{total_rows:,} rows (batch {batch_num})...")
                    
                    ok, msg = upload_to_rfq_sheet(res_df, sheet_name.strip(), batch_size=500, progress_callback=update_progress)
                    
                    upload_progress.progress(1.0)
                    upload_status.empty()
                    
                    if ok: 
                        st.success(msg)
                        # Popup button to open the sheet directly
                        st.link_button("🔗 Open Target Sheet", rfq_sheet_url, type="primary")
                    else: 
                        st.error(msg)
        else:
            # Fallback if RFQ_url is not configured
            st.warning("RFQ_url not configured in secrets. Cloud upload unavailable.")
            st.download_button(
                "📥 Download Results CSV", 
                res_df.to_csv(index=False), 
                "pricing_results.csv", 
                "text/csv", 
                type="secondary",
                key="download_results_csv",
                use_container_width=True
            )
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
        # Security Check
        if check_admin_access("dist"):
            st.success("🔓 Access Granted")
            
            # 1. Dashboard Metrics
            # Fetch data first to populate metrics
            resolved = get_resolved_distances_from_sheet()
            failed = get_failed_distances_from_sheet()
            
            st.markdown("### 📊 Status Dashboard")
            m1, m2, m3 = st.columns(3)
            m1.metric("Ready to Import", len(resolved), help="Distances found in Sheet ready to be added")
            m2.metric("Failed Lookups", len(failed), delta_color="inverse" if len(failed) > 0 else "off", help="Routes Google couldn't find")
            m3.metric("Current Pickle Size", f"{len(DISTANCE_MATRIX):,}", help="Total distances currently in memory")
            
            st.markdown("---")
            
            # 2. Action Buttons (Primary Actions)
            c1, c2 = st.columns([1, 1])
            with c1:
                import_clicked = st.button("🔄 Import Distances to Pickle", type="primary", disabled=len(resolved)==0, use_container_width=True)
            with c2:
                refresh_clicked = st.button("🔍 Refresh Sheet Data", use_container_width=True)

            if refresh_clicked:
                st.rerun()

            # 3. Import Logic (Preserved)
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

            # 4. Detailed Data Views (Tabs)
            if resolved or failed:
                st.markdown("#### 📋 Details")
                t1, t2 = st.tabs(["✅ Ready to Import", "⚠️ Failed Lookups"])
                
                with t1:
                    if resolved:
                        st.caption("These distances will be added to your local database.")
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
                        st.warning("These routes failed API lookup. Please manually enter distances in Column G of the Sheet.")
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

            # 5. Permanent Save Section
            st.markdown("---")
            st.subheader("💾 Save Changes Permanently")
            
            col_info, col_dl = st.columns([2, 1])
            with col_info:
                st.info("updates are lost on reboot. Download the updated `.pkl` and commit it to Git.")
            
            with col_dl:
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
