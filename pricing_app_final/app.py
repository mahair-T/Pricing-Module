import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import pickle
import os
import io
import re
from datetime import datetime

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="Freight Pricing Tool", page_icon="🚚", layout="wide")

APP_DIR = os.path.dirname(os.path.abspath(__file__))

def normalize_english_text(text):
    """Normalize English text for lookups - lowercase, strip, normalize whitespace/hyphens."""
    if pd.isna(text) or text is None:
        return None
    text = str(text).strip().lower()
    text = re.sub(r'[-_]+', ' ', text)  # Replace hyphens/underscores with space
    text = re.sub(r'\s+', ' ', text)    # Collapse multiple spaces
    return text

# ============================================
# LOAD CITY NORMALIZATION (The Source of Truth)
# ============================================
@st.cache_resource
def load_city_normalization():
    """
    Load city normalization from CSV.
    Builds:
    1. Variant -> Canonical (Arabic) mapping.
    2. Canonical (Arabic) -> English Display Name mapping.
    3. Region mappings.
    """
    norm_path = os.path.join(APP_DIR, 'model_export', 'city_normalization_updated.csv')
    
    variant_to_canonical = {}
    variant_to_canonical_lower = {}
    variant_to_region = {}
    canonical_to_region = {}
    canonical_to_english = {}  # Map Canonical Arabic -> Best English Name
    
    if os.path.exists(norm_path):
        try:
            df = pd.read_csv(norm_path)
            
            # 1. First Pass: Build basic mappings
            for _, row in df.iterrows():
                variant = str(row['variant']).strip()
                canonical = str(row['canonical']).strip()
                region = row['region'] if pd.notna(row.get('region')) else None

                # Map variant to canonical
                variant_to_canonical[variant] = canonical
                
                # Lowercase lookup
                variant_lower = normalize_english_text(variant)
                if variant_lower:
                    variant_to_canonical_lower[variant_lower] = canonical
                
                # Region mappings
                if region:
                    variant_to_region[variant] = region
                    if canonical not in canonical_to_region:
                        canonical_to_region[canonical] = region
            
            # 2. Second Pass: Determine best English display name for each Canonical
            # We group by canonical and look for an English variant
            grouped = df.groupby('canonical')['variant'].apply(list)
            
            for canonical, variants in grouped.items():
                english_name = None
                
                # Heuristic: Find first variant that is ASCII (English)
                # Prefer "Title Case" over "ALL CAPS" if possible
                ascii_variants = [v for v in variants if re.match(r'^[A-Za-z\s\-\(\)\.]+$', str(v))]
                
                if ascii_variants:
                    # Try to find a Title Case one (e.g., 'Jeddah' vs 'JEDDAH')
                    title_case = [v for v in ascii_variants if str(v)[0].isupper() and not str(v).isupper()]
                    if title_case:
                        english_name = title_case[0]
                    else:
                        english_name = ascii_variants[0]
                
                # Fallback: If no English variant found, use the Canonical Arabic itself
                canonical_to_english[canonical] = english_name if english_name else canonical

        except Exception as e:
            st.error(f"Error loading city normalization file: {e}")
            return {}, {}, {}, {}, {}
            
    return variant_to_canonical, variant_to_canonical_lower, variant_to_region, canonical_to_region, canonical_to_english

# Load mappings
CITY_TO_CANONICAL, CITY_TO_CANONICAL_LOWER, CITY_TO_REGION, CANONICAL_TO_REGION, CITY_AR_TO_EN = load_city_normalization()

# Create Reverse Mapping (English Display -> Arabic Canonical)
CITY_EN_TO_AR = {v: k for k, v in CITY_AR_TO_EN.items()}

# ============================================
# HELPER FUNCTIONS (City & Normalization)
# ============================================
def normalize_city(city_raw):
    """Normalize city name to standard Arabic canonical form."""
    if pd.isna(city_raw) or city_raw == '':
        return None, False
    city = str(city_raw).strip()
    
    # 1. Exact Match
    if city in CITY_TO_CANONICAL:
        return CITY_TO_CANONICAL[city], True
    
    # 2. Lowercase Match
    city_lower = normalize_english_text(city)
    if city_lower and city_lower in CITY_TO_CANONICAL_LOWER:
        return CITY_TO_CANONICAL_LOWER[city_lower], True
    
    # 3. Fallback: Check if it is already a known English display name
    if city in CITY_EN_TO_AR:
        return CITY_EN_TO_AR[city], True

    return city, False

def get_city_region(city):
    """Get region for a city."""
    # Try canonical mapping first (most reliable)
    canonical = CITY_TO_CANONICAL.get(city, city)
    if canonical in CANONICAL_TO_REGION:
        return CANONICAL_TO_REGION[canonical]
    # Try variant mapping
    if city in CITY_TO_REGION:
        return CITY_TO_REGION[city]
    return None

def to_english_city(city_ar):
    """Convert Canonical Arabic -> English Display Name."""
    if city_ar in CITY_AR_TO_EN:
        return CITY_AR_TO_EN[city_ar]
    norm, found = normalize_city(city_ar)
    if found and norm in CITY_AR_TO_EN:
        return CITY_AR_TO_EN[norm]
    return city_ar

def to_arabic_city(city_en):
    """Convert English Display Name -> Canonical Arabic."""
    if city_en in CITY_EN_TO_AR:
        return CITY_EN_TO_AR[city_en]
    norm, found = normalize_city(city_en)
    return norm if norm else city_en

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

def log_exception(exception_type, details):
    """Log exception to Google Sheets."""
    if 'error_log' not in st.session_state:
        st.session_state.error_log = []
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'type': exception_type,
        **details
    }
    st.session_state.error_log.append(log_entry)
    
    try:
        worksheet = get_error_sheet()
        if worksheet:
            row = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                exception_type,
                details.get('pickup_city', details.get('original_value', '')),
                details.get('destination_city', details.get('normalized_to', '')),
                details.get('pickup_en', ''),
                details.get('destination_en', ''),
                str(details)
            ]
            worksheet.append_row(row)
    except Exception as e:
        pass

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
    'Unknown': 'Unknown',
    'أكسيد الحديد': 'Iron Oxide',
    'أكياس ورقية فارغة': 'Empty Paper Bags',
    'أنابيب': 'Pipes',
    'اجهزه كهربائيه': 'Electrical Equipment',
    'ارز': 'Rice',
    'اسمدة': 'Fertilizer',
    'الطرود': 'Parcels',
    'براميل': 'Barrels',
    'بلاستيك': 'Plastic',
    'تمر': 'Dates',
    'جبس': 'Gypsum',
    'خردة': 'Scrap',
    'خشب': 'Wood',
    'رمل': 'Sand',
    'رمل السيليكا': 'Silica Sand',
    'زجاج ليفي': 'Fiberglass',
    'زيوت': 'Oils',
    'سكر': 'Sugar',
    'سلع إستهلاكية': 'Consumer Goods',
    'سيراميك': 'Ceramics',
    'فحم': 'Coal',
    'فوارغ': 'Empties',
    'كابلات': 'Cables',
    'كيماوي': 'Chemicals',
    'لفات حديد': 'Steel Coils',
    'معدات': 'Equipment',
    'منتجات الصلب': 'Steel Products',
    'مواد بناء': 'Building Materials',
    'مياه': 'Water',
    'ورق': 'Paper',
}
COMMODITY_AR = {v: k for k, v in COMMODITY_EN.items()}

def to_english_commodity(commodity_ar):
    return COMMODITY_EN.get(commodity_ar, commodity_ar)

def to_arabic_commodity(commodity_en):
    return COMMODITY_AR.get(commodity_en, commodity_en)

# ============================================
# INDEX + SHRINKAGE MODEL (Rare Lane Model)
# ============================================
class IndexShrinkagePredictor:
    """Index + Shrinkage model for lanes with some historical data."""
    
    def __init__(self, model_artifacts):
        m = model_artifacts
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
    
    def _predict_index(self, lane):
        if lane in self.lane_multipliers:
            return self.current_index * self.lane_multipliers[lane]
        return None
    
    def _predict_shrinkage(self, lane, pickup_city, dest_city):
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
            lane_mean = stats['lane_mean']
            lane_n = stats['lane_n']
            lam = lane_n / (lane_n + self.k)
            return lam * lane_mean + (1 - lam) * city_prior
        else:
            canonical_lane = self.get_canonical_lane(lane)
            if canonical_lane in self.lane_stats:
                stats = self.lane_stats[canonical_lane]
                lane_mean = stats['lane_mean']
                lane_n = stats['lane_n']
                lam = lane_n / (lane_n + self.k)
                return lam * lane_mean + (1 - lam) * city_prior
            return city_prior
    
    def has_lane_data(self, lane):
        if lane in self.lane_stats or lane in self.lane_multipliers:
            return True
        canonical_lane = self.get_canonical_lane(lane)
        return canonical_lane in self.lane_stats or canonical_lane in self.lane_multipliers
    
    def get_canonical_lane(self, lane):
        parts = lane.split(' → ')
        if len(parts) == 2:
            pickup, dest = parts
            p_can = CITY_TO_CANONICAL.get(pickup, pickup)
            d_can = CITY_TO_CANONICAL.get(dest, dest)
            return f"{p_can} → {d_can}"
        return lane
    
    def predict(self, pickup_city, dest_city, distance_km=None):
        lane = f"{pickup_city} → {dest_city}"
        canonical_lane = self.get_canonical_lane(lane)
        lookup_lane = canonical_lane if (canonical_lane in self.lane_stats or canonical_lane in self.lane_multipliers) else lane
        
        pickup_canonical = CITY_TO_CANONICAL.get(pickup_city, pickup_city)
        dest_canonical = CITY_TO_CANONICAL.get(dest_city, dest_city)
        
        idx_pred = self._predict_index(lookup_lane)
        shrink_pred = self._predict_shrinkage(lookup_lane, pickup_canonical, dest_canonical)
        
        if idx_pred is not None and shrink_pred is not None:
            predicted_cpk = (idx_pred + shrink_pred) / 2
            method = 'Index + Shrinkage'
            if lane in self.lane_stats:
                n = self.lane_stats[lane]['lane_n']
                confidence = 'High' if n >= 20 else 'Medium' if n >= 5 else 'Low'
            else:
                confidence = 'Low'
        elif shrink_pred is not None:
            predicted_cpk = shrink_pred
            method = 'Shrinkage'
            confidence = 'Medium'
        else:
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
# BLEND MODEL (New Lane Model - 0.7 Regional)
# ============================================
class BlendPredictor:
    """Blend model (0.7 Regional + 0.3 City) for completely new lanes."""
    
    def __init__(self, model_artifacts):
        m = model_artifacts
        self.config = m['config']
        self.regional_weight = self.config.get('regional_weight', 0.7)
        self.current_index = m['current_index']
        self.pickup_city_mult = m['pickup_city_mult']
        self.dest_city_mult = m['dest_city_mult']
        self.city_to_region = m['city_to_region']
        self.regional_cpk = m['regional_cpk']
        self.model_date = self.config.get('training_date', 'Unknown')
    
    def predict(self, pickup_city, dest_city, distance_km=None):
        p_region = CANONICAL_TO_REGION.get(pickup_city) or self.city_to_region.get(pickup_city)
        d_region = CANONICAL_TO_REGION.get(dest_city) or self.city_to_region.get(dest_city)
        
        if not p_region:
            p_can = CITY_TO_CANONICAL.get(pickup_city, pickup_city)
            p_region = CANONICAL_TO_REGION.get(p_can)
        if not d_region:
            d_can = CITY_TO_CANONICAL.get(dest_city, dest_city)
            d_region = CANONICAL_TO_REGION.get(d_can)
        
        regional_cpk = None
        if p_region and d_region:
            regional_cpk = self.regional_cpk.get((p_region, d_region))
        
        p_can = CITY_TO_CANONICAL.get(pickup_city, pickup_city)
        d_can = CITY_TO_CANONICAL.get(dest_city, dest_city)
        
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
VALID_CITIES_AR = set(df_knn['pickup_city'].unique()) | set(df_knn['destination_city'].unique())

# ============================================
# BACKHAUL PROBABILITY CALCULATION
# ============================================
@st.cache_data
def calculate_city_cpk_stats():
    city_stats = {}
    outbound = df_knn.groupby('pickup_city').agg({
        'total_carrier_price': 'median',
        'distance': 'median'
    }).reset_index()
    outbound['outbound_cpk'] = outbound['total_carrier_price'] / outbound['distance']
    
    inbound = df_knn.groupby('destination_city').agg({
        'total_carrier_price': 'median',
        'distance': 'median'
    }).reset_index()
    inbound['inbound_cpk'] = inbound['total_carrier_price'] / inbound['distance']
    
    for _, row in outbound.iterrows():
        city = row['pickup_city']
        if city not in city_stats: city_stats[city] = {}
        city_stats[city]['outbound_cpk'] = row['outbound_cpk']
    
    for _, row in inbound.iterrows():
        city = row['destination_city']
        if city not in city_stats: city_stats[city] = {}
        city_stats[city]['inbound_cpk'] = row['inbound_cpk']
    
    return city_stats

CITY_CPK_STATS = calculate_city_cpk_stats()

def get_backhaul_probability(dest_city):
    stats = CITY_CPK_STATS.get(dest_city, {})
    inbound_cpk = stats.get('inbound_cpk')
    outbound_cpk = stats.get('outbound_cpk')
    
    if inbound_cpk is None or outbound_cpk is None or outbound_cpk == 0:
        return 'Unknown', MARGIN_UNKNOWN, None
    
    ratio = inbound_cpk / outbound_cpk
    
    if ratio >= BACKHAUL_HIGH_THRESHOLD:
        return 'High', MARGIN_HIGH_BACKHAUL, ratio
    elif ratio >= BACKHAUL_MEDIUM_THRESHOLD:
        return 'Medium', MARGIN_MEDIUM_BACKHAUL, ratio
    else:
        return 'Low', MARGIN_LOW_BACKHAUL, ratio

# ============================================
# DISTANCE & PRICING HELPERS
# ============================================
def calculate_rental_cost(distance_km):
    if not distance_km or distance_km <= 0:
        return None
    days = distance_km / RENTAL_KM_PER_DAY
    days_rounded = 1.0 if days < 1 else round(days * 2) / 2
    return round(days_rounded * RENTAL_COST_PER_DAY, 0)

def calculate_reference_sell(pickup_ar, dest_ar, vehicle_ar, lane_data, recommended_sell):
    recent_data = lane_data[lane_data['days_ago'] <= RECENCY_WINDOW] if len(lane_data) > 0 else pd.DataFrame()
    
    if len(recent_data) > 0 and 'total_shipper_price' in recent_data.columns:
        recent_sell = recent_data['total_shipper_price'].dropna()
        if len(recent_sell) > 0:
            median_sell = recent_sell.median()
            if median_sell > 0:
                return round(median_sell, 0), 'Recent 90d'
    
    if len(lane_data) > 0 and 'total_shipper_price' in lane_data.columns:
        hist_sell = lane_data['total_shipper_price'].dropna()
        if len(hist_sell) > 0:
            hist_median = hist_sell.median()
            if hist_median > 0:
                # Apply index adjustment if available
                if 'MONTHLY_INDEX' in config and 'pickup_date' in lane_data.columns:
                    try:
                        lane_data_copy = lane_data.copy()
                        lane_data_copy['pickup_date'] = pd.to_datetime(lane_data_copy['pickup_date'])
                        lane_data_copy['month'] = lane_data_copy['pickup_date'].dt.to_period('M').astype(str)
                        monthly_index = config.get('MONTHLY_INDEX', {})
                        current_index = config.get('CURRENT_INDEX', 1.8)
                        valid_months = lane_data_copy['month'].value_counts()
                        if len(valid_months) > 0:
                            weighted_index = sum(monthly_index.get(m, current_index) * c for m, c in valid_months.items())
                            total_weight = sum(valid_months.values)
                            avg_hist_index = weighted_index / total_weight if total_weight > 0 else 0
                            index_ratio = current_index / avg_hist_index if avg_hist_index > 0 else 1.0
                            return round(hist_median * index_ratio, 0), 'Historical (idx-adj)'
                    except: pass
                return round(hist_median, 0), 'Historical'
    
    if recommended_sell:
        return recommended_sell, 'Recommended'
    
    return None, 'N/A'

def get_distance(pickup_ar, dest_ar, lane_data=None):
    lane = f"{pickup_ar} → {dest_ar}"
    reverse_lane = f"{dest_ar} → {pickup_ar}"
    
    p_can = CITY_TO_CANONICAL.get(pickup_ar, pickup_ar)
    d_can = CITY_TO_CANONICAL.get(dest_ar, dest_ar)
    
    if lane_data is not None and len(lane_data) > 0:
        dist_vals = lane_data[lane_data['distance'] > 0]['distance']
        if len(dist_vals) > 0: return dist_vals.median(), 'Historical'
    
    reverse_data = df_knn[df_knn['lane'] == reverse_lane]
    if len(reverse_data) > 0:
        dist_vals = reverse_data[reverse_data['distance'] > 0]['distance']
        if len(dist_vals) > 0: return dist_vals.median(), 'Historical (reverse)'
    
    if DISTANCE_LOOKUP.get(lane): return DISTANCE_LOOKUP[lane], 'Historical'
    if DISTANCE_LOOKUP.get(reverse_lane): return DISTANCE_LOOKUP[reverse_lane], 'Historical (reverse)'
    
    if (pickup_ar, dest_ar) in DISTANCE_MATRIX: return DISTANCE_MATRIX[(pickup_ar, dest_ar)], 'Matrix'
    if (p_can, d_can) in DISTANCE_MATRIX: return DISTANCE_MATRIX[(p_can, d_can)], 'Matrix (canonical)'
    if (dest_ar, pickup_ar) in DISTANCE_MATRIX: return DISTANCE_MATRIX[(dest_ar, pickup_ar)], 'Matrix (reverse)'
    if (d_can, p_can) in DISTANCE_MATRIX: return DISTANCE_MATRIX[(d_can, p_can)], 'Matrix (canonical reverse)'
    
    log_exception('missing_distance', {
        'pickup': pickup_ar, 'dest': dest_ar, 
        'pickup_en': to_english_city(pickup_ar), 'dest_en': to_english_city(dest_ar)
    })
    return 0, 'Missing'

def round_to_nearest(value, nearest):
    if value is None or pd.isna(value): return None
    return int(round(value / nearest) * nearest)

# ============================================
# PRICE CASCADE
# ============================================
def calculate_prices(pickup_ar, dest_ar, vehicle_ar, distance_km, lane_data=None):
    lane = f"{pickup_ar} → {dest_ar}"
    
    if lane_data is None:
        lane_data = df_knn[
            (df_knn['lane'] == lane) & 
            (df_knn['vehicle_type'] == vehicle_ar)
        ].copy()
    
    recent_data = lane_data[lane_data['days_ago'] <= RECENCY_WINDOW] if len(lane_data) > 0 else pd.DataFrame()
    recent_count = len(recent_data)
    
    buy_price = None
    model_used = None
    confidence = None
    
    if recent_count >= 1:
        buy_price = recent_data['total_carrier_price'].median()
        model_used = 'Recency'
        confidence = 'High' if recent_count >= 5 else 'Medium' if recent_count >= 2 else 'Low'
    elif index_shrink_predictor and index_shrink_predictor.has_lane_data(lane):
        pred = index_shrink_predictor.predict(pickup_ar, dest_ar, distance_km)
        buy_price = pred.get('predicted_cost')
        model_used = pred['method']
        confidence = pred['confidence']
    elif blend_predictor:
        pred = blend_predictor.predict(pickup_ar, dest_ar, distance_km)
        buy_price = pred.get('predicted_cost')
        model_used = pred['method']
        confidence = pred['confidence']
    else:
        if distance_km and distance_km > 0:
            buy_price = distance_km * 1.8
            model_used = 'Default CPK'
            confidence = 'Very Low'
    
    buy_price_rounded = round_to_nearest(buy_price, 100)
    backhaul_prob, margin, backhaul_ratio = get_backhaul_probability(dest_ar)
    
    sell_price_rounded = round_to_nearest(buy_price_rounded * (1 + margin), 50) if buy_price_rounded else None
    ref_sell, ref_sell_source = calculate_reference_sell(pickup_ar, dest_ar, vehicle_ar, lane_data, sell_price_rounded)
    ref_sell_rounded = round_to_nearest(ref_sell, 50) if ref_sell else sell_price_rounded
    
    return {
        'buy_price': buy_price_rounded,
        'sell_price': sell_price_rounded,
        'ref_sell_price': ref_sell_rounded,
        'ref_sell_source': ref_sell_source,
        'rental_cost': calculate_rental_cost(distance_km),
        'target_margin': f"{margin:.0%}",
        'backhaul_probability': backhaul_prob,
        'backhaul_ratio': round(backhaul_ratio, 2) if backhaul_ratio else None,
        'model_used': model_used,
        'confidence': confidence,
        'recent_count': recent_count,
    }

# ============================================
# AMMUNITION & BULK
# ============================================
def select_spaced_samples(df, n_samples):
    if len(df) == 0: return pd.DataFrame()
    df = df.sort_values('days_ago')
    selected = []
    last_days_ago = -MIN_DAYS_APART
    
    # Try recent first
    for _, row in df[df['days_ago'] <= RECENT_PRIORITY_DAYS].iterrows():
        if row['days_ago'] >= last_days_ago + MIN_DAYS_APART:
            selected.append(row)
            last_days_ago = row['days_ago']
            if len(selected) >= n_samples: break
            
    # Then older
    if len(selected) < n_samples:
        for _, row in df[df['days_ago'] > RECENT_PRIORITY_DAYS].iterrows():
            if row['days_ago'] >= last_days_ago + MIN_DAYS_APART:
                selected.append(row)
                last_days_ago = row['days_ago']
                if len(selected) >= n_samples: break
                
    return pd.DataFrame(selected) if selected else pd.DataFrame()

def get_ammunition_loads(lane, vehicle_ar, commodity=None):
    matches = df_knn[
        (df_knn['lane'] == lane) & 
        (df_knn['vehicle_type'] == vehicle_ar) &
        (df_knn['days_ago'] <= MAX_AGE_DAYS)
    ].copy()
    
    if len(matches) == 0: return pd.DataFrame(), pd.DataFrame()
    
    if commodity and commodity not in ['', 'Auto', 'auto', None]:
        same = matches[matches['commodity'] == commodity]
        other = matches[matches['commodity'] != commodity]
    else:
        most_common = matches['commodity'].mode().iloc[0] if len(matches) > 0 else None
        same = matches[matches['commodity'] == most_common] if most_common else pd.DataFrame()
        other = matches[matches['commodity'] != most_common] if most_common else pd.DataFrame()
    
    return select_spaced_samples(same, 5), select_spaced_samples(other, 5)

def price_single_route(pickup_ar, dest_ar, vehicle_ar=None, commodity=None, weight=None):
    if vehicle_ar is None or vehicle_ar in ['', 'Auto', 'auto', None]:
        vehicle_ar = DEFAULT_VEHICLE_AR
    
    lane = f"{pickup_ar} → {dest_ar}"
    lane_data = df_knn[(df_knn['lane'] == lane) & (df_knn['vehicle_type'] == vehicle_ar)].copy()
    
    if commodity and commodity not in ['', 'Auto', 'auto', None]:
        lane_commodity_data = lane_data[lane_data['commodity'] == commodity]
        if len(lane_commodity_data) > 0:
            lane_data = lane_commodity_data
    
    recent_data = lane_data[lane_data['days_ago'] <= RECENCY_WINDOW]
    
    # Buy price (carrier) stats
    hist_count = len(lane_data) if len(lane_data) > 0 else 0
    hist_min = lane_data['total_carrier_price'].min() if hist_count > 0 else None
    hist_max = lane_data['total_carrier_price'].max() if hist_count > 0 else None
    hist_median = lane_data['total_carrier_price'].median() if hist_count > 0 else None
    
    recent_count = len(recent_data) if len(recent_data) > 0 else 0
    recent_min = recent_data['total_carrier_price'].min() if recent_count > 0 else None
    recent_max = recent_data['total_carrier_price'].max() if recent_count > 0 else None
    recent_median = recent_data['total_carrier_price'].median() if recent_count > 0 else None
    
    # Sell price (shipper) stats
    hist_sell_min = lane_data['total_shipper_price'].min() if hist_count > 0 else None
    hist_sell_max = lane_data['total_shipper_price'].max() if hist_count > 0 else None
    hist_sell_median = lane_data['total_shipper_price'].median() if hist_count > 0 else None
    
    recent_sell_min = recent_data['total_shipper_price'].min() if recent_count > 0 else None
    recent_sell_max = recent_data['total_shipper_price'].max() if recent_count > 0 else None
    recent_sell_median = recent_data['total_shipper_price'].median() if recent_count > 0 else None
    
    if commodity is None or commodity in ['', 'Auto', 'auto']:
        commodity = lane_data['commodity'].mode().iloc[0] if len(lane_data) > 0 else df_knn['commodity'].mode().iloc[0]
    
    if weight is None or weight == 0:
        comm_weights = df_knn[(df_knn['commodity'] == commodity) & (df_knn['weight'] > 0)]['weight']
        weight = comm_weights.median() if len(comm_weights) > 0 else df_knn['weight'].median()
    
    # Get distance
    distance, distance_source = get_distance(pickup_ar, dest_ar, lane_data)
    if distance == 0:
        distance = 500
        distance_source = 'Default'
    
    # Calculate prices using cascade
    pricing = calculate_prices(pickup_ar, dest_ar, vehicle_ar, distance, lane_data)
    
    # Determine if rare lane
    is_rare_lane = recent_count == 0
    
    result = {
        'Vehicle_Type': to_english_vehicle(vehicle_ar),
        'Commodity': to_english_commodity(commodity),
        'Weight_Tons': round(weight, 1),
        'Distance_km': round(distance, 0),
        'Distance_Source': distance_source,
        'Hist_Count': hist_count,
        'Hist_Min': round(hist_min, 0) if hist_min else None,
        'Hist_Median': round(hist_median, 0) if hist_median else None,
        'Hist_Max': round(hist_max, 0) if hist_max else None,
        f'Recent_{RECENCY_WINDOW}d_Count': recent_count,
        f'Recent_{RECENCY_WINDOW}d_Min': round(recent_min, 0) if recent_min else None,
        f'Recent_{RECENCY_WINDOW}d_Median': round(recent_median, 0) if recent_median else None,
        f'Recent_{RECENCY_WINDOW}d_Max': round(recent_max, 0) if recent_max else None,
        # Sell price (shipper) stats
        'Hist_Sell_Min': round(hist_sell_min, 0) if hist_sell_min else None,
        'Hist_Sell_Median': round(hist_sell_median, 0) if hist_sell_median else None,
        'Hist_Sell_Max': round(hist_sell_max, 0) if hist_sell_max else None,
        f'Recent_{RECENCY_WINDOW}d_Sell_Min': round(recent_sell_min, 0) if recent_sell_min else None,
        f'Recent_{RECENCY_WINDOW}d_Sell_Median': round(recent_sell_median, 0) if recent_sell_median else None,
        f'Recent_{RECENCY_WINDOW}d_Sell_Max': round(recent_sell_max, 0) if recent_sell_max else None,
        'Buy_Price': pricing['buy_price'],
        'Rec_Sell_Price': pricing['sell_price'],  # Recommended sell (buy + margin)
        'Ref_Sell_Price': pricing['ref_sell_price'],  # Reference sell (market data)
        'Ref_Sell_Source': pricing['ref_sell_source'],
        'Rental_Cost': pricing['rental_cost'],
        'Target_Margin': pricing['target_margin'],
        'Backhaul_Probability': pricing['backhaul_probability'],
        'Backhaul_Ratio': pricing['backhaul_ratio'],
        'Model_Used': pricing['model_used'],
        'Confidence': pricing['confidence'],
        'Cost_Per_KM': round(pricing['buy_price'] / distance, 2) if pricing['buy_price'] and distance > 0 else None,
        'Is_Rare_Lane': is_rare_lane,
    }
    
    # Add model predictions for comparison
    if index_shrink_predictor:
        idx_pred = index_shrink_predictor.predict(pickup_ar, dest_ar, distance)
        result['IndexShrink_Price'] = idx_pred.get('predicted_cost')
        result['IndexShrink_Upper'] = idx_pred.get('cost_high')
        result['IndexShrink_Method'] = idx_pred['method']
    
    if blend_predictor:
        blend_pred = blend_predictor.predict(pickup_ar, dest_ar, distance)
        result['Blend_Price'] = blend_pred.get('predicted_cost')
        result['Blend_Upper'] = blend_pred.get('cost_high')
        result['Blend_Method'] = blend_pred['method']
    
    return result

def lookup_route_stats(pickup_ar, dest_ar, vehicle_ar=None):
    if not vehicle_ar or vehicle_ar in ['', 'Auto', 'auto']: vehicle_ar = DEFAULT_VEHICLE_AR
    lane_data = df_knn[(df_knn['lane'] == f"{pickup_ar} → {dest_ar}") & (df_knn['vehicle_type'] == vehicle_ar)].copy()
    distance, _ = get_distance(pickup_ar, dest_ar, lane_data)
    pricing = calculate_prices(pickup_ar, dest_ar, vehicle_ar, distance, lane_data)
    return {
        'From': to_english_city(pickup_ar),
        'To': to_english_city(dest_ar),
        'Distance': int(distance) if distance else 0,
        'Buy_Price': pricing['buy_price'],
        'Rec_Sell': pricing['sell_price'],
        'Ref_Sell': pricing['ref_sell_price'],
        'Ref_Sell_Src': pricing['ref_sell_source'],
        'Rental_Cost': pricing['rental_cost'],
        'Margin': pricing['target_margin'],
        'Model': pricing['model_used'],
        'Confidence': pricing['confidence'],
        'Recent_N': pricing['recent_count'],
    }

# ============================================
# DROPDOWN OPTIONS
# ============================================
# Build dropdowns from the Loaded City Map (Source of Truth)
# Get all valid canonical cities
all_canonicals = sorted(list(set(CITY_TO_CANONICAL.values())))
# Convert to English for the UI
pickup_cities_en = sorted(list(set([to_english_city(c) for c in all_canonicals])))
dest_cities_en = pickup_cities_en  # Same list for destination

vehicle_types_en = sorted(set(VEHICLE_TYPE_EN.values()))
commodities = sorted(set([to_english_commodity(c) for c in df_knn['commodity'].unique()]))

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

# TAB 1: SINGLE ROUTE
with tab1:
    st.subheader("📋 Route Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        # Default selection logic
        def_idx = pickup_cities_en.index('Jeddah') if 'Jeddah' in pickup_cities_en else 0
        pickup_en = st.selectbox("Pickup City", options=pickup_cities_en, index=def_idx, key='single_pickup')
        pickup_city = to_arabic_city(pickup_en)

    with col2:
        def_idx = dest_cities_en.index('Riyadh') if 'Riyadh' in dest_cities_en else 0
        dest_en = st.selectbox("Destination City", options=dest_cities_en, index=def_idx, key='single_dest')
        destination_city = to_arabic_city(dest_en)

    with col3:
        def_idx = vehicle_types_en.index(DEFAULT_VEHICLE_EN) if DEFAULT_VEHICLE_EN in vehicle_types_en else 0
        vehicle_en = st.selectbox("Vehicle Type", options=vehicle_types_en, index=def_idx, key='single_vehicle')
        vehicle_type = to_arabic_vehicle(vehicle_en)

    st.subheader("📦 Optional Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.selectbox("Container", options=['Auto-detect', 'Yes (Container)', 'No (Non-container)'], key='single_container')

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
        
        # Save for report
        st.session_state.last_result = result
        st.session_state.last_lane = {'pickup_ar': pickup_city, 'dest_ar': destination_city, 'pickup_en': pickup_en, 'dest_en': dest_en}
        
        # Display Logic
        if result['Distance_km'] == 0 or result['Distance_Source'] == 'Default':
            st.error(f"⚠️ Distance data missing or estimated ({result['Distance_km']} km)")
        
        st.header("💰 Pricing Results")
        st.info(f"**{pickup_en} → {dest_en}** | 🚛 {result['Vehicle_Type']} | 📏 {result['Distance_km']:.0f} km | ⚖️ {result['Weight_Tons']:.1f} T")
        
        # Metrics
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
        
        if result['Is_Rare_Lane']:
            st.warning(f"⚠️ **Rare Lane** - No recent loads. Using model prediction.")

        # History Tables
        st.markdown("---")
        st.subheader("📊 Price History")
        hc, rc = result['Hist_Count'], result[f'Recent_{RECENCY_WINDOW}d_Count']
        
        if hc > 0 or rc > 0:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Historical")
                st.dataframe(pd.DataFrame({
                    'Metric': ['Count', 'Min', 'Median', 'Max'],
                    'Buy Price': [f"{hc} loads" if hc else "—", f"{result['Hist_Min']:,} SAR" if result['Hist_Min'] else "—", f"{result['Hist_Median']:,} SAR" if result['Hist_Median'] else "—", f"{result['Hist_Max']:,} SAR" if result['Hist_Max'] else "—"],
                    'Sell Price': [f"{hc} loads" if hc else "—", f"{result['Hist_Sell_Min']:,} SAR" if result['Hist_Sell_Min'] else "—", f"{result['Hist_Sell_Median']:,} SAR" if result['Hist_Sell_Median'] else "—", f"{result['Hist_Sell_Max']:,} SAR" if result['Hist_Sell_Max'] else "—"]
                }), use_container_width=True, hide_index=True)
            with c2:
                st.markdown(f"### Recent ({RECENCY_WINDOW}d)")
                st.dataframe(pd.DataFrame({
                    'Metric': ['Count', 'Min', 'Median', 'Max'],
                    'Buy Price': [f"{rc} loads" if rc else "—", f"{result[f'Recent_{RECENCY_WINDOW}d_Min']:,} SAR" if result[f'Recent_{RECENCY_WINDOW}d_Min'] else "—", f"{result[f'Recent_{RECENCY_WINDOW}d_Median']:,} SAR" if result[f'Recent_{RECENCY_WINDOW}d_Median'] else "—", f"{result[f'Recent_{RECENCY_WINDOW}d_Max']:,} SAR" if result[f'Recent_{RECENCY_WINDOW}d_Max'] else "—"],
                    'Sell Price': [f"{rc} loads" if rc else "—", f"{result[f'Recent_{RECENCY_WINDOW}d_Sell_Min']:,} SAR" if result[f'Recent_{RECENCY_WINDOW}d_Sell_Min'] else "—", f"{result[f'Recent_{RECENCY_WINDOW}d_Sell_Median']:,} SAR" if result[f'Recent_{RECENCY_WINDOW}d_Sell_Median'] else "—", f"{result[f'Recent_{RECENCY_WINDOW}d_Sell_Max']:,} SAR" if result[f'Recent_{RECENCY_WINDOW}d_Sell_Max'] else "—"]
                }), use_container_width=True, hide_index=True)
        else:
            st.warning("No historical or recent data available")
        
        # Model Comparison (Restored)
        st.markdown("---")
        with st.expander("🔮 Model Predictions Comparison", expanded=False):
            model_df = []
            if 'IndexShrink_Price' in result and result['IndexShrink_Price']:
                model_df.append({
                    'Model': result['IndexShrink_Method'],
                    'Prediction': f"{result['IndexShrink_Price']:,.0f} SAR",
                    'Upper Bound': f"{result['IndexShrink_Upper']:,.0f} SAR",
                })
            if 'Blend_Price' in result and result['Blend_Price']:
                model_df.append({
                    'Model': result['Blend_Method'],
                    'Prediction': f"{result['Blend_Price']:,.0f} SAR",
                    'Upper Bound': f"{result['Blend_Upper']:,.0f} SAR",
                })
            if model_df:
                st.dataframe(pd.DataFrame(model_df), use_container_width=True, hide_index=True)
            
        # Ammunition
        if not result['Is_Rare_Lane']:
            st.markdown("---")
            st.subheader("🚚 Your Ammunition (Recent Matches)")
            same, other = get_ammunition_loads(f"{pickup_city} → {destination_city}", vehicle_type, comm_in)
            
            if len(same) > 0:
                st.markdown(f"**Same Commodity ({comm_in if comm_in else result['Commodity']}):**")
                disp = same[['pickup_date', 'total_carrier_price', 'days_ago']].copy()
                disp.columns = ['Date', 'Carrier (SAR)', 'Days Ago']
                disp['Carrier (SAR)'] = disp['Carrier (SAR)'].round(0).astype(int)
                st.dataframe(disp, use_container_width=True, hide_index=True)
            
            if len(other) > 0:
                st.markdown("**Other Commodities:**")
                disp = other[['pickup_date', 'commodity', 'total_carrier_price', 'days_ago']].copy()
                disp['commodity'] = disp['commodity'].apply(to_english_commodity)
                disp.columns = ['Date', 'Commodity', 'Carrier (SAR)', 'Days Ago']
                disp['Carrier (SAR)'] = disp['Carrier (SAR)'].round(0).astype(int)
                st.dataframe(disp, use_container_width=True, hide_index=True)

    # Report Button
    if 'last_result' in st.session_state and 'last_lane' in st.session_state:
        st.markdown("---")
        with st.expander("🚨 Report Issue", expanded=False):
            st.caption(f"Reporting: {st.session_state.last_lane['pickup_en']} → {st.session_state.last_lane['dest_en']}")
            iss = st.selectbox("Issue Type", ["Distance incorrect", "Distance 0/missing", "Data wrong", "Other"], key='rep_type')
            note = st.text_area("Notes", key='rep_note')
            if st.button("Submit Report"):
                l, r = st.session_state.last_lane, st.session_state.last_result
                if report_distance_issue(l['pickup_ar'], l['dest_ar'], l['pickup_en'], l['dest_en'], r['Distance_km'], iss, note):
                    st.success("Report submitted")
                else: st.warning("Report failed (GS not configured)")

# TAB 2: BULK
with tab2:
    st.subheader("📦 Bulk Route Lookup")
    st.download_button("📥 Template", pd.DataFrame({'From':['Jeddah','Riyadh'], 'To':['Riyadh','Dammam'], 'Vehicle_Type':['Flatbed Trailer','']}).to_csv(index=False), "template.csv", "text/csv")
    
    upl = st.file_uploader("Upload CSV", type=['csv'])
    if upl:
        try:
            r_df = pd.read_csv(upl)
            if st.button("🔍 Process"):
                res, unmat = [], []
                prog = st.progress(0)
                
                for i, row in r_df.iterrows():
                    p_raw, d_raw = str(row.get('From','')).strip(), str(row.get('To','')).strip()
                    p_ar, p_ok = normalize_city(p_raw)
                    d_ar, d_ok = normalize_city(d_raw)
                    
                    if not p_ok: 
                        unmat.append({'Row':i+1, 'Col':'From', 'Val':p_raw})
                        log_exception('unmatched_city', {'row':i+1, 'col':'From', 'val':p_raw})
                    if not d_ok: 
                        unmat.append({'Row':i+1, 'Col':'To', 'Val':d_raw})
                        log_exception('unmatched_city', {'row':i+1, 'col':'To', 'val':d_raw})
                    
                    v_raw = str(row.get('Vehicle_Type','')).strip()
                    v_ar = to_arabic_vehicle(v_raw) if v_raw not in ['','nan','None','Auto'] else DEFAULT_VEHICLE_AR
                    
                    res.append(lookup_route_stats(p_ar, d_ar, v_ar))
                    prog.progress((i+1)/len(r_df))
                
                res_df = pd.DataFrame(res)
                st.dataframe(res_df, use_container_width=True, hide_index=True)
                st.download_button("📥 Download Results", res_df.to_csv(index=False), "results.csv", "text/csv", type="primary")
                
                if unmat:
                    st.warning(f"{len(unmat)} unmatched cities")
                    st.dataframe(pd.DataFrame(unmat))
        except Exception as e: st.error(f"Error: {e}")

# EXCEPTION LOG
st.markdown("---")
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
