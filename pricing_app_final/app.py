import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import pickle
import os
import io
import re
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
    """
    try:
        worksheet = get_matched_distances_sheet()
        if not worksheet:
            return []
        
        all_data = worksheet.get_all_values()
        resolved = []
        
        for i, row in enumerate(all_data[1:], start=2):  # Skip header, track row number
            if len(row) >= 9:
                pickup_ar = row[1]
                dest_ar = row[2]
                distance_value = row[6]  # Column G - Distance_Value
                status = row[7]
                added_to_pickle = row[8]
                
                # Only get rows with valid numeric distance that haven't been added yet
                if distance_value and added_to_pickle != 'Yes':
                    try:
                        # Handle various number formats
                        dist_clean = str(distance_value).replace(',', '').replace(' km', '').strip()
                        dist_km = float(dist_clean)
                        if dist_km > 0:
                            resolved.append({
                                'row': i,
                                'pickup_ar': pickup_ar,
                                'dest_ar': dest_ar,
                                'distance_km': dist_km
                            })
                    except (ValueError, TypeError):
                        pass
        
        return resolved
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
        rows_to_mark = []
        
        for item in resolved:
            key = (item['pickup_ar'], item['dest_ar'])
            rev_key = (item['dest_ar'], item['pickup_ar'])
            
            # Add if not already in matrix
            if key not in distance_matrix:
                distance_matrix[key] = item['distance_km']
                added_count += 1
            
            # Also add reverse direction
            if rev_key not in distance_matrix:
                distance_matrix[rev_key] = item['distance_km']
            
            rows_to_mark.append(item['row'])
        
        # Save updated pickle
        with open(pkl_path, 'wb') as f:
            pickle.dump(distance_matrix, f)
        
        # Update global variable
        DISTANCE_MATRIX = distance_matrix
        
        # Mark rows as added in sheet
        mark_distances_as_added(rows_to_mark)
        
        return True, added_count, f"Added {added_count} new distances to pickle"
    
    except Exception as e:
        return False, 0, f"Error updating pickle: {str(e)}"

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
    - region: Geographic region (Eastern, Western, Central, etc.)
    
    Returns multiple lookup dictionaries for flexible matching.
    """
    norm_path = os.path.join(APP_DIR, 'model_export', 'city_normalization_with_regions.csv')
    
    variant_to_canonical = {}           # Exact match lookup
    variant_to_canonical_lower = {}     # Lowercase lookup for English names
    variant_to_canonical_aggressive = {} # Aggressive normalization (no spaces/punctuation)
    variant_to_region = {}              # Variant -> Region mapping
    canonical_to_region = {}            # Canonical -> Region mapping
    canonical_to_english = {}           # Canonical Arabic -> English display name
    
    if os.path.exists(norm_path):
        try:
            df = pd.read_csv(norm_path)
            
            for _, row in df.iterrows():
                variant = str(row['variant']).strip()
                canonical = str(row['canonical']).strip()
                region = row['region'] if pd.notna(row.get('region')) else None

                variant_to_canonical[variant] = canonical
                
                # Also add lowercase version for case-insensitive English matching
                variant_lower = normalize_english_text(variant)
                if variant_lower:
                    variant_to_canonical_lower[variant_lower] = canonical
                
                # Add aggressive normalized version (strips all spaces/punctuation)
                variant_aggressive = normalize_aggressive(variant)
                if variant_aggressive:
                    variant_to_canonical_aggressive[variant_aggressive] = canonical
                
                if region:
                    variant_to_region[variant] = region
                    if canonical not in canonical_to_region:
                        canonical_to_region[canonical] = region
            
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
            variant_to_region, canonical_to_region, canonical_to_english)

# Load mappings
(CITY_TO_CANONICAL, CITY_TO_CANONICAL_LOWER, CITY_TO_CANONICAL_AGGRESSIVE,
 CITY_TO_REGION, CANONICAL_TO_REGION, CITY_AR_TO_EN) = load_city_normalization()
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
    """Get region for a city (checks both variant and canonical mappings)."""
    if city in CITY_TO_REGION:
        return CITY_TO_REGION[city]
    if city in CANONICAL_TO_REGION:
        return CANONICAL_TO_REGION[city]
    # Try normalizing first then looking up
    canonical = CITY_TO_CANONICAL.get(city, city)
    if canonical in CANONICAL_TO_REGION:
        return CANONICAL_TO_REGION[canonical]
    return None

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
# BLEND MODEL (New Lane Model - 0.7 Regional)
# Used for completely NEW lanes with NO historical data
# Blends regional averages with city-level multipliers
# ============================================
class BlendPredictor:
    """
    Blend model (0.7 Regional + 0.3 City) for completely new lanes.
    
    For lanes with zero historical data, we estimate price by:
    1. Regional CPK: Average CPK for the region pair (e.g., Eastern → Western)
    2. City Multipliers: Pickup and destination city adjustment factors
    
    Final: 70% Regional + 30% City-adjusted
    
    This provides a reasonable estimate even for never-seen routes by
    leveraging geographic patterns in freight pricing.
    """
    
    def __init__(self, model_artifacts):
        m = model_artifacts
        self.config = m['config']
        self.regional_weight = self.config.get('regional_weight', 0.7)  # 70% regional
        self.current_index = m['current_index']
        self.pickup_city_mult = m['pickup_city_mult']   # City-specific adjustments
        self.dest_city_mult = m['dest_city_mult']
        self.city_to_region = m['city_to_region']       # City → Region mapping
        self.regional_cpk = m['regional_cpk']           # (Region, Region) → CPK
        self.model_date = self.config.get('training_date', 'Unknown')
    
    def predict(self, pickup_city, dest_city, distance_km=None):
        # Get regions
        p_can = CITY_TO_CANONICAL.get(pickup_city, pickup_city)
        d_can = CITY_TO_CANONICAL.get(dest_city, dest_city)
        
        p_region = CANONICAL_TO_REGION.get(p_can) or self.city_to_region.get(p_can)
        d_region = CANONICAL_TO_REGION.get(d_can) or self.city_to_region.get(d_can)
        
        # Fallback if no region found
        if not p_region:
            p_region = CANONICAL_TO_REGION.get(p_can)
        if not d_region:
            d_region = CANONICAL_TO_REGION.get(d_can)
        
        regional_cpk = None
        if p_region and d_region:
            regional_cpk = self.regional_cpk.get((p_region, d_region))
        
        # City Multipliers
        p_mult = self.pickup_city_mult.get(pickup_city, self.pickup_city_mult.get(p_can, 1.0))
        d_mult = self.dest_city_mult.get(dest_city, self.dest_city_mult.get(d_can, 1.0))
        city_cpk = p_mult * d_mult * self.current_index
        
        if regional_cpk is not None:
            predicted_cpk = self.regional_weight * regional_cpk + (1 - self.regional_weight) * city_cpk
            method = f'Blend ({self.regional_weight:.0%} Regional)'
            confidence = 'Low'
        else:
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

# Set of all cities with historical trip data (used to filter distance lookups)
# Missing distances are only logged to MatchedDistances if at least one city is in this set
VALID_CITIES_AR = set(df_knn['pickup_city'].unique()) | set(df_knn['destination_city'].unique())

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

def get_distance(pickup_ar, dest_ar, lane_data=None, immediate_log=False):
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
    # ONLY if:
    # 1. We have valid English names for both cities
    # 2. AT LEAST ONE city has historical data (exists in VALID_CITIES_AR)
    if pickup_en and dest_en and pickup_en != pickup_ar and dest_en != dest_ar:
        # Check if at least one city (or their canonical forms) has historical data
        pickup_has_history = pickup_ar in VALID_CITIES_AR or p_can in VALID_CITIES_AR
        dest_has_history = dest_ar in VALID_CITIES_AR or d_can in VALID_CITIES_AR
        
        if pickup_has_history or dest_has_history:
            log_missing_distance_for_lookup(pickup_ar, dest_ar, pickup_en, dest_en, immediate=immediate_log)
    
    return 0, 'Missing'

def round_to_nearest(value, nearest):
    """Round value to nearest multiple (e.g., 100 for buy price, 50 for sell price)."""
    return int(round(value / nearest) * nearest) if value and not pd.isna(value) else None

def calculate_prices(pickup_ar, dest_ar, vehicle_ar, distance_km, lane_data=None):
    """
    Calculate buy/sell prices using the pricing cascade.
    
    PRICING CASCADE (in order of preference):
    1. RECENCY: If recent loads exist (within RECENCY_WINDOW days), use median
    2. INDEX+SHRINKAGE: If historical data exists, use Index+Shrinkage model
    3. BLEND: If no lane data, use regional blend model (70% regional + 30% city)
    4. DEFAULT: Last resort - use default CPK × distance
    
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
        # 2. INDEX + SHRINKAGE: Lane has historical data but no recent loads
        p = index_shrink_predictor.predict(pickup_ar, dest_ar, distance_km)
        buy_price, model, conf = p.get('predicted_cost'), p['method'], p['confidence']
    elif blend_predictor:
        # 3. BLEND MODEL: Completely new lane, use regional + city estimates
        p = blend_predictor.predict(pickup_ar, dest_ar, distance_km)
        buy_price, model, conf = p.get('predicted_cost'), p['method'], p['confidence']
    else:
        # 4. DEFAULT: Last resort fallback
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

def lookup_route_stats(pickup_ar, dest_ar, vehicle_ar=None):
    """Lookup route stats for bulk pricing - uses batch logging (immediate_log=False)."""
    if not vehicle_ar or vehicle_ar in ['', 'Auto', 'auto']: vehicle_ar = DEFAULT_VEHICLE_AR
    lane_data = df_knn[(df_knn['lane'] == f"{pickup_ar} → {dest_ar}") & (df_knn['vehicle_type'] == vehicle_ar)].copy()
    # Use immediate_log=False for bulk pricing (queues errors for batch write)
    dist, _ = get_distance(pickup_ar, dest_ar, lane_data, immediate_log=False)
    pricing = calculate_prices(pickup_ar, dest_ar, vehicle_ar, dist, lane_data)
    return {
        'From': to_english_city(pickup_ar), 'To': to_english_city(dest_ar), 'Distance': int(dist) if dist else 0,
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
if blend_predictor: model_status.append("✅ Blend 0.7")
dist_status = f"✅ {len(DISTANCE_MATRIX):,} distances" if DISTANCE_MATRIX else ""
st.caption(f"ML-powered pricing | Domestic | {' | '.join(model_status)} | {dist_status}")

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
        with st.expander("🚨 Report Issue", expanded=False):
            st.caption(f"Reporting: {st.session_state.last_lane['pickup_en']} → {st.session_state.last_lane['dest_en']}")
            iss = st.selectbox("Issue Type", ["Distance incorrect", "Distance 0/missing", "Data wrong", "Other"], key='rep_type')
            note = st.text_area("Notes", key='rep_note')
            if st.button("Submit Report"):
                l, r = st.session_state.last_lane, st.session_state.last_result
                if report_distance_issue(l['pickup_ar'], l['dest_ar'], l['pickup_en'], l['dest_en'], r['Distance_km'], iss, note): st.success("Report submitted")
                else: st.warning("Report failed (GS not configured)")

with tab2:
    st.subheader("📦 Bulk Route Lookup")
    
    st.markdown(f"""
    **Upload a CSV to get pricing for each route.**
    
    **Required:** `From`, `To`  
    **Optional:** `Vehicle_Type` (default: Flatbed Trailer)
    
    **Output columns:**
    - Distance, Buy Price (rounded to 100), Sell Price (rounded to 50)
    - Target Margin, Backhaul Probability
    - Model Used, Confidence, Recent Count
    """)
    st.download_button("📥 Template", pd.DataFrame({'From':['Jeddah','Riyadh'], 'To':['Riyadh','Dammam'], 'Vehicle_Type':['Flatbed Trailer','']}).to_csv(index=False), "template.csv", "text/csv")
    
    upl = st.file_uploader("Upload CSV", type=['csv'])
    if upl:
        try:
            r_df = pd.read_csv(upl)
            st.success(f"✅ Loaded {len(r_df)} routes")
            
            with st.expander("📋 Preview"):
                st.dataframe(r_df.head(10), use_container_width=True)
            
            if st.button("🔍 Look Up All Routes", type="primary", use_container_width=True):
                res, unmat = [], []
                prog = st.progress(0)
                status_text = st.empty()
                for i, row in r_df.iterrows():
                    p_raw, d_raw = str(row.get('From','')).strip(), str(row.get('To','')).strip()
                    p_ar, p_ok = normalize_city(p_raw)
                    d_ar, d_ok = normalize_city(d_raw)
                    if not p_ok: 
                        unmat.append({'Row':i+1, 'Col':'From', 'Val':p_raw})
                        log_exception('unmatched_city', {'row':i+1, 'column':'From', 'original_value':p_raw})
                    if not d_ok: 
                        unmat.append({'Row':i+1, 'Col':'To', 'Val':d_raw})
                        log_exception('unmatched_city', {'row':i+1, 'column':'To', 'original_value':d_raw})
                    v_raw = str(row.get('Vehicle_Type','')).strip()
                    v_ar = to_arabic_vehicle(v_raw) if v_raw not in ['','nan','None','Auto'] else DEFAULT_VEHICLE_AR
                    res.append(lookup_route_stats(p_ar, d_ar, v_ar))
                    prog.progress((i+1)/len(r_df))
                    status_text.text(f"Looking up {i+1}/{len(r_df)}: {p_raw} → {d_raw}")
                status_text.text("✅ Complete!")
                st.session_state.bulk_results = pd.DataFrame(res)
                st.session_state.bulk_unmatched = pd.DataFrame(unmat)
                
                # Flush any pending error logs to Google Sheets in one batch
                flushed_ok, flushed_count = flush_error_log_to_sheet()
                if flushed_count > 0:
                    st.caption(f"📝 Logged {flushed_count} exceptions to error sheet")
                
                # Flush any pending matched distances to Google Sheets
                dist_ok, dist_count = flush_matched_distances_to_sheet()
                if dist_count > 0:
                    st.caption(f"📏 Logged {dist_count} missing distances for Google Maps lookup")
        except Exception as e: st.error(f"Error: {e}")

    # Results Display (Outside indentation so it persists)
    if 'bulk_results' in st.session_state and not st.session_state.bulk_results.empty:
        res_df = st.session_state.bulk_results
        st.markdown("---")
        st.subheader("📊 Results")
        st.dataframe(res_df, use_container_width=True, hide_index=True)
        st.download_button("📥 Download Results", res_df.to_csv(index=False), "results.csv", "text/csv", type="primary")
        
        st.markdown("---")
        st.subheader("☁️ Cloud Upload")
        st.caption(f"Target Sheet: `{BULK_PRICING_SHEET_URL}`")
        if st.button("☁️ Upload Results to Google Sheet"):
            upload_progress = st.progress(0)
            upload_status = st.empty()
            
            def update_progress(rows_done, total_rows, batch_num):
                upload_progress.progress(rows_done / total_rows)
                upload_status.text(f"Uploaded {rows_done:,}/{total_rows:,} rows (batch {batch_num})...")
            
            ok, msg = upload_to_gsheet(res_df, batch_size=500, progress_callback=update_progress)
            
            upload_progress.progress(1.0)
            upload_status.empty()
            if ok: st.success(msg)
            else: st.error(msg)

    if 'bulk_unmatched' in st.session_state and not st.session_state.bulk_unmatched.empty:
        st.warning(f"{len(st.session_state.bulk_unmatched)} unmatched cities")
        st.dataframe(st.session_state.bulk_unmatched)

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
                        batch_results.append(lookup_route_stats(p_ar, d_ar, DEFAULT_VEHICLE_AR))
                    
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
        st.info("""
        **How this works:**
        1. Missing distances are logged to the 'MatchedDistances' sheet
        2. Google Apps Script uses GOOGLEMAPS_DISTANCE formula to get distances
        3. Once distances resolve, click below to import them into the app
        """)
        
        # Show count of pending distances
        resolved = get_resolved_distances_from_sheet()
        if resolved:
            st.success(f"✅ {len(resolved)} new distances ready to import")
            
            # Preview
            with st.expander("Preview distances to import"):
                preview_df = pd.DataFrame([
                    {'From': to_english_city(r['pickup_ar']), 
                     'To': to_english_city(r['dest_ar']), 
                     'Distance (km)': r['distance_km']}
                    for r in resolved[:20]
                ])
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
                if len(resolved) > 20:
                    st.caption(f"... and {len(resolved) - 20} more")
        else:
            st.caption("No new distances available to import")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Import Distances to Pickle", type="primary", disabled=len(resolved)==0):
                with st.spinner("Importing distances..."):
                    success, count, message = update_distance_pickle_from_sheet()
                    if success:
                        st.success(message)
                        if count > 0:
                            st.balloons()
                            st.cache_resource.clear()  # Clear cached data to reload
                    else:
                        st.error(message)
        
        with col2:
            if st.button("🔍 Refresh Count"):
                st.rerun()

st.markdown("---")
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
