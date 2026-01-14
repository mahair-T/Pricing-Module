import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import pickle
import os
import io
import re


def normalize_english_text(text):
    """Normalize English text for lookups - lowercase, strip, normalize whitespace/hyphens."""
    if pd.isna(text) or text is None:
        return None
    text = str(text).strip().lower()
    # Normalize hyphens and multiple spaces
    text = re.sub(r'[-_]+', ' ', text)  # Replace hyphens/underscores with space
    text = re.sub(r'\s+', ' ', text)    # Collapse multiple spaces
    return text

st.set_page_config(page_title="Freight Pricing Tool", page_icon="🚚", layout="wide")

APP_DIR = os.path.dirname(os.path.abspath(__file__))

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
    from datetime import datetime
    
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
    from datetime import datetime
    
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

# Error bar percentages by confidence level
ERROR_BARS = {
    'High': 0.10,
    'Medium': 0.15,
    'Low': 0.25,
    'Very Low': 0.35
}

# Margin settings based on backhaul probability
MARGIN_HIGH_BACKHAUL = 0.12      # 12% margin when good backhaul
MARGIN_MEDIUM_BACKHAUL = 0.18   # 18% margin
MARGIN_LOW_BACKHAUL = 0.25      # 25% margin when poor backhaul
MARGIN_UNKNOWN = 0.20           # 20% default when no data

# Backhaul thresholds (inbound/outbound CPK ratio)
BACKHAUL_HIGH_THRESHOLD = 0.85   # Ratio > 0.85 = High backhaul prob
BACKHAUL_MEDIUM_THRESHOLD = 0.65 # Ratio 0.65-0.85 = Medium
# Below 0.65 = Low backhaul probability

# ============================================
# LOAD CITY NORMALIZATION WITH REGIONS
# ============================================
@st.cache_resource
def load_city_normalization():
    """Load city normalization with regions."""
    norm_path = os.path.join(APP_DIR, 'model_export', 'city_normalization_with_regions.csv')
    if os.path.exists(norm_path):
        df = pd.read_csv(norm_path)
        # Build lookup dicts
        variant_to_canonical = {}
        variant_to_canonical_lower = {}  # Lowercase lookup for English names
        variant_to_region = {}
        canonical_to_region = {}
        
        for _, row in df.iterrows():
            variant = row['variant']
            canonical = row['canonical']
            
            variant_to_canonical[variant] = canonical
            
            # Also add lowercase version for case-insensitive matching
            variant_lower = normalize_english_text(variant)
            if variant_lower:
                variant_to_canonical_lower[variant_lower] = canonical
            
            if pd.notna(row.get('region')):
                variant_to_region[variant] = row['region']
                canonical_to_region[canonical] = row['region']
        
        return variant_to_canonical, variant_to_canonical_lower, variant_to_region, canonical_to_region
    return {}, {}, {}, {}

CITY_TO_CANONICAL, CITY_TO_CANONICAL_LOWER, CITY_TO_REGION, CANONICAL_TO_REGION = load_city_normalization()

# ============================================
# CITY NORMALIZATION - Comprehensive
# ============================================
CITY_NORMALIZE = {
    # English to Arabic mappings
    'Jeddah': 'جدة', 'Jiddah': 'جدة', 'Jedda': 'جدة', 'jeddah': 'جدة',
    'Riyadh': 'الرياض', 'Riyad': 'الرياض', 'riyadh': 'الرياض',
    'Dammam': 'الدمام', 'Damam': 'الدمام', 'dammam': 'الدمام',
    'Makkah': 'مكة المكرمة', 'Mecca': 'مكة المكرمة', 'Mekkah': 'مكة المكرمة',
    'Madinah': 'المدينة المنورة', 'Madina': 'المدينة المنورة', 'Medina': 'المدينة المنورة',
    'Yanbu': 'ينبع', 'Yenbu': 'ينبع',
    'Rabigh': 'رابغ',
    'Tabuk': 'تبوك', 'Tabouk': 'تبوك',
    'Taif': 'الطائف', 'Tayef': 'الطائف',
    'Jubail': 'الجبيل', 'Al Jubail': 'الجبيل', 'Al-Jubail': 'الجبيل',
    'Al Hasa': 'الْأَحْسَاء', 'Al-Hasa': 'الْأَحْسَاء', 'Ahsa': 'الْأَحْسَاء', 'Hofuf': 'الْأَحْسَاء',
    'Al Kharj': 'الخرج', 'Al-Kharij': 'الخرج', 'Kharj': 'الخرج',
    'Qassim': 'القصيم', 'Al Qassim': 'القصيم', 'Al-Qassim': 'القصيم', 'Qaseem': 'القصيم',
    'Al Baha': 'ٱلْبَاحَة', 'Baha': 'ٱلْبَاحَة', 'Al-Baha': 'ٱلْبَاحَة',
    'Jazan': 'جازان', 'Jizan': 'جازان', 'Gizan': 'جازان',
    'Najran': 'نجران', 'Nejran': 'نجران',
    'Arar': 'عرعر',
    'Skaka': 'سكاكا', 'Sakaka': 'سكاكا',
    'Hafar Al Batin': 'حفر الباطن', 'Hafr Al Batin': 'حفر الباطن',
    'Sudair': 'سدير',
    'Hail': 'حَائِل', 'Haail': 'حَائِل',
    'Neom': 'تبوك', 'NEOM': 'تبوك',
    'Buraidah': 'القصيم', 'Buraydah': 'القصيم',
    'Al Khobar': 'Al-Khobar',
    'Duba': 'ضبا',
    'Abha': 'Abha',
    'Khamis Mushait': 'Khamis Mushait', 'Khamis': 'Khamis Mushait',
    'Umluj': 'Umluj',
    'Al-Muzahmiyya': 'Al-Muzahmiyya',
    'Muhayil': 'Muhayil',
    'Bisha': 'Bisha',
    'Al Ula': 'Al Ula',
    'Sharma': 'Sharma',
    'Haql': 'Haql',
    'Dumah Al Jandal': 'Dumah Al Jandal',
    'Duwadmi': 'Duwadmi',
    'Wadi Ad dawaser': 'Wadi Ad dawaser', 'Wadi Aldawaser': 'Wadi Ad dawaser',
    'Dahban': 'Dahban',
    'Beljurashi': 'Beljurashi',
    'Khafji': 'Khafji',
    'Al Lith': 'Al Lith',
    'Buqaiq': 'Buqaiq',
    'Al Qunfudah': 'Al Qunfudah',
    'AMAALA': 'AMAALA',
    'Ranyah': 'Ranyah',
    'Al-Jumum': 'Al-Jumum',
    'Nairyah': 'Nairyah',
    'Al Farwaniyah': 'Al Farwaniyah',
    'Haradh': 'Haradh',
    'Thuwwal': 'Thuwwal',
    'Tanajib': 'Tanajib',
    'Al Qatif': 'Al Qatif',
    'Murjan': 'Murjan',
    "Al Majma'ah": "Al Majma'ah",
    # Arabic variations
    'جدة': 'جدة', 'جده': 'جدة',
    'الرياض': 'الرياض', 'رياض': 'الرياض',
    'الدمام': 'الدمام', 'دمام': 'الدمام',
    'مكة': 'مكة المكرمة', 'مكه': 'مكة المكرمة', 'مكة المكرمة': 'مكة المكرمة',
    'المدينة': 'المدينة المنورة', 'المدينه': 'المدينة المنورة', 'المدينة المنورة': 'المدينة المنورة',
    'الطائف': 'الطائف', 'طائف': 'الطائف',
    'تبوك': 'تبوك',
    'القصيم': 'القصيم', 'قصيم': 'القصيم',
    'الخرج': 'الخرج', 'خرج': 'الخرج',
    'جازان': 'جازان', 'جيزان': 'جازان',
    'نجران': 'نجران',
    'عرعر': 'عرعر',
    'سكاكا': 'سكاكا',
    'ينبع': 'ينبع',
    'رابغ': 'رابغ',
    'الجبيل': 'الجبيل',
    'الاحساء': 'الْأَحْسَاء', 'الأحساء': 'الْأَحْسَاء', 'الْأَحْسَاء': 'الْأَحْسَاء',
    'الباحة': 'ٱلْبَاحَة', 'ٱلْبَاحَة': 'ٱلْبَاحَة',
    'حفر الباطن': 'حفر الباطن',
    'سدير': 'سدير',
    'حَائِل': 'حَائِل', 'حائل': 'حَائِل',
    'ابها': 'Abha', 'أبها': 'Abha',
    'خميس مشيط': 'Khamis Mushait',
    'طريف': 'طريف',
    'ضرما': 'ضرما',
    'راس الخير': 'راس الخير',
    'الملك عبدالله': 'الملك عبدالله',
    'شقراء': 'شقراء',
    'عسير': 'عسير',
    'ضبا': 'ضبا',
    'الخرمة': 'الخرمة',
    'الشعيبة': 'الشعيبة',
}

# English display names
CITY_EN = {
    'جدة': 'Jeddah',
    'الرياض': 'Riyadh',
    'الدمام': 'Dammam',
    'مكة المكرمة': 'Makkah',
    'المدينة المنورة': 'Madinah',
    'الطائف': 'Taif',
    'تبوك': 'Tabuk',
    'القصيم': 'Qassim',
    'الخرج': 'Al Kharj',
    'جازان': 'Jazan',
    'نجران': 'Najran',
    'عرعر': 'Arar',
    'سكاكا': 'Skaka',
    'ينبع': 'Yanbu',
    'رابغ': 'Rabigh',
    'الْأَحْسَاء': 'Al Hasa',
    'ٱلْبَاحَة': 'Al Baha',
    'حفر الباطن': 'Hafar Al Batin',
    'سدير': 'Sudair',
    'حَائِل': 'Hail',
    'الجبيل': 'Jubail',
    'طريف': 'Turaif',
    'ضرما': 'Dirma',
    'راس الخير': 'Ras Al Khair',
    'الملك عبدالله': 'King Abdullah',
    'شقراء': 'Shaqra',
    'عسير': 'Asir',
    'ضبا': 'Duba',
    'الخرمة': 'Al Khurmah',
    'الشعيبة': 'Al Shuaiba',
    'Abha': 'Abha',
    'Khamis Mushait': 'Khamis Mushait',
    'Umluj': 'Umluj',
    'Al-Muzahmiyya': 'Al-Muzahmiyya',
    'Muhayil': 'Muhayil',
    'Bisha': 'Bisha',
    'Al Ula': 'Al Ula',
    'Sharma': 'Sharma',
    'Haql': 'Haql',
    'Dumah Al Jandal': 'Dumah Al Jandal',
    'Duwadmi': 'Duwadmi',
    'Wadi Ad dawaser': 'Wadi Ad dawaser',
    'Dahban': 'Dahban',
    'Beljurashi': 'Beljurashi',
    'Khafji': 'Khafji',
    'Al Lith': 'Al Lith',
    'Buqaiq': 'Buqaiq',
    'Al Qunfudah': 'Al Qunfudah',
    'AMAALA': 'AMAALA',
    'Ranyah': 'Ranyah',
    'Al-Jumum': 'Al-Jumum',
    'Nairyah': 'Nairyah',
    'Al Farwaniyah': 'Al Farwaniyah',
    'Haradh': 'Haradh',
    'Thuwwal': 'Thuwwal',
    'Tanajib': 'Tanajib',
    'Al Qatif': 'Al Qatif',
    'Al-Khobar': 'Al-Khobar',
    'Murjan': 'Murjan',
    "Al Majma'ah": "Al Majma'ah",
}

CITY_AR = {v: k for k, v in CITY_EN.items()}

def normalize_city(city_raw):
    """Normalize city name to standard form."""
    if pd.isna(city_raw) or city_raw == '':
        return None, False
    city = str(city_raw).strip()
    
    # Check city normalization file first (exact match)
    if city in CITY_TO_CANONICAL:
        return CITY_TO_CANONICAL[city], True
    
    # Try lowercase/normalized lookup (handles "JEDDAH", "jeddah", etc.)
    city_lower = normalize_english_text(city)
    if city_lower and city_lower in CITY_TO_CANONICAL_LOWER:
        return CITY_TO_CANONICAL_LOWER[city_lower], True
    
    # Direct lookup in hardcoded dict
    if city in CITY_NORMALIZE:
        return CITY_NORMALIZE[city], True
    
    # Case-insensitive lookup in hardcoded dict
    for key, val in CITY_NORMALIZE.items():
        if key.lower() == city.lower():
            return val, True
    
    # Partial match (last resort)
    for key, val in CITY_NORMALIZE.items():
        if key.lower() in city.lower() or city.lower() in key.lower():
            return val, True
    
    return city, False

def get_city_region(city):
    """Get region for a city (checks both variant and canonical mappings)."""
    # First try variant mapping
    if city in CITY_TO_REGION:
        return CITY_TO_REGION[city]
    # Then try canonical mapping
    if city in CANONICAL_TO_REGION:
        return CANONICAL_TO_REGION[city]
    # Try normalizing first then looking up
    canonical = CITY_TO_CANONICAL.get(city, city)
    if canonical in CANONICAL_TO_REGION:
        return CANONICAL_TO_REGION[canonical]
    return None

def to_english_city(city_ar):
    return CITY_EN.get(city_ar, city_ar)

def to_arabic_city(city_en):
    normalized, _ = normalize_city(city_en)
    return normalized if normalized else city_en

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
        # Try to get priors with canonical names as fallback
        p_prior = self.pickup_priors.get(pickup_city)
        if p_prior is None:
            pickup_canonical = CITY_TO_CANONICAL.get(pickup_city, pickup_city)
            p_prior = self.pickup_priors.get(pickup_canonical, self.global_mean)
        
        d_prior = self.dest_priors.get(dest_city)
        if d_prior is None:
            dest_canonical = CITY_TO_CANONICAL.get(dest_city, dest_city)
            d_prior = self.dest_priors.get(dest_canonical, self.global_mean)
        
        city_prior = (p_prior + d_prior) / 2
        
        # Try lane stats with canonical lane as fallback
        if lane in self.lane_stats:
            stats = self.lane_stats[lane]
            lane_mean = stats['lane_mean']
            lane_n = stats['lane_n']
            lam = lane_n / (lane_n + self.k)
            return lam * lane_mean + (1 - lam) * city_prior
        else:
            # Try canonical lane
            canonical_lane = self.get_canonical_lane(lane)
            if canonical_lane in self.lane_stats:
                stats = self.lane_stats[canonical_lane]
                lane_mean = stats['lane_mean']
                lane_n = stats['lane_n']
                lam = lane_n / (lane_n + self.k)
                return lam * lane_mean + (1 - lam) * city_prior
            return city_prior
    
    def has_lane_data(self, lane):
        """Check if we have data for this lane (tries canonical names too)."""
        if lane in self.lane_stats or lane in self.lane_multipliers:
            return True
        
        # Try with canonical names (using lowercase lookup)
        canonical_lane = self.get_canonical_lane(lane)
        if canonical_lane != lane:
            return canonical_lane in self.lane_stats or canonical_lane in self.lane_multipliers
        return False
    
    def get_canonical_lane(self, lane):
        """Get the canonical version of a lane string."""
        parts = lane.split(' → ')
        if len(parts) == 2:
            pickup, dest = parts
            
            # Try exact match first, then lowercase
            pickup_canonical = CITY_TO_CANONICAL.get(pickup)
            if pickup_canonical is None:
                pickup_lower = normalize_english_text(pickup)
                pickup_canonical = CITY_TO_CANONICAL_LOWER.get(pickup_lower, pickup) if pickup_lower else pickup
            
            dest_canonical = CITY_TO_CANONICAL.get(dest)
            if dest_canonical is None:
                dest_lower = normalize_english_text(dest)
                dest_canonical = CITY_TO_CANONICAL_LOWER.get(dest_lower, dest) if dest_lower else dest
            
            return f"{pickup_canonical} → {dest_canonical}"
        return lane
    
    def predict(self, pickup_city, dest_city, distance_km=None):
        lane = f"{pickup_city} → {dest_city}"
        
        # Try canonical lane if original doesn't exist
        canonical_lane = self.get_canonical_lane(lane)
        lookup_lane = canonical_lane if (canonical_lane in self.lane_stats or canonical_lane in self.lane_multipliers) else lane
        
        # Also get canonical city names for priors
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
        # Get regions - use CSV-based CANONICAL_TO_REGION as source of truth
        # First try the global mapping, fall back to model's mapping
        p_region = CANONICAL_TO_REGION.get(pickup_city) or self.city_to_region.get(pickup_city)
        d_region = CANONICAL_TO_REGION.get(dest_city) or self.city_to_region.get(dest_city)
        
        # If still no region, try normalizing the city name first
        if not p_region:
            pickup_canonical = CITY_TO_CANONICAL.get(pickup_city, pickup_city)
            p_region = CANONICAL_TO_REGION.get(pickup_canonical)
        if not d_region:
            dest_canonical = CITY_TO_CANONICAL.get(dest_city, dest_city)
            d_region = CANONICAL_TO_REGION.get(dest_canonical)
        
        # Get regional CPK
        regional_cpk = None
        if p_region and d_region:
            regional_cpk = self.regional_cpk.get((p_region, d_region))
        
        # Get city multipliers (try canonical names for lookup)
        pickup_canonical = CITY_TO_CANONICAL.get(pickup_city)
        if pickup_canonical is None:
            pickup_lower = normalize_english_text(pickup_city)
            pickup_canonical = CITY_TO_CANONICAL_LOWER.get(pickup_lower, pickup_city) if pickup_lower else pickup_city
        
        dest_canonical = CITY_TO_CANONICAL.get(dest_city)
        if dest_canonical is None:
            dest_lower = normalize_english_text(dest_city)
            dest_canonical = CITY_TO_CANONICAL_LOWER.get(dest_lower, dest_city) if dest_lower else dest_city
        
        p_mult = self.pickup_city_mult.get(pickup_city, self.pickup_city_mult.get(pickup_canonical, 1.0))
        d_mult = self.dest_city_mult.get(dest_city, self.dest_city_mult.get(dest_canonical, 1.0))
        city_cpk = p_mult * d_mult * self.current_index
        
        # Blend
        if regional_cpk is not None:
            predicted_cpk = self.regional_weight * regional_cpk + (1 - self.regional_weight) * city_cpk
            method = f'Blend ({self.regional_weight:.0%} Regional)'
            confidence = 'Low'  # New lane model is always Low confidence
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
    
    # CatBoost model is optional (not used in main pricing cascade)
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
    
    # Load Blend model (new lane model)
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
    """Calculate inbound and outbound CPK for each city for backhaul estimation."""
    city_stats = {}
    
    # Calculate outbound CPK (city as pickup)
    outbound = df_knn.groupby('pickup_city').agg({
        'total_carrier_price': 'median',
        'distance': 'median'
    }).reset_index()
    outbound['outbound_cpk'] = outbound['total_carrier_price'] / outbound['distance']
    
    # Calculate inbound CPK (city as destination)
    inbound = df_knn.groupby('destination_city').agg({
        'total_carrier_price': 'median',
        'distance': 'median'
    }).reset_index()
    inbound['inbound_cpk'] = inbound['total_carrier_price'] / inbound['distance']
    
    # Merge
    for _, row in outbound.iterrows():
        city = row['pickup_city']
        if city not in city_stats:
            city_stats[city] = {}
        city_stats[city]['outbound_cpk'] = row['outbound_cpk']
    
    for _, row in inbound.iterrows():
        city = row['destination_city']
        if city not in city_stats:
            city_stats[city] = {}
        city_stats[city]['inbound_cpk'] = row['inbound_cpk']
    
    return city_stats

CITY_CPK_STATS = calculate_city_cpk_stats()

def get_backhaul_probability(dest_city):
    """
    Determine backhaul probability based on inbound/outbound CPK ratio.
    High ratio = good backhaul availability = lower margin needed.
    """
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
# DISTANCE LOOKUP FUNCTION
# ============================================
def calculate_rental_cost(distance_km):
    """Calculate rental cost index based on distance.
    Assumes 800 SAR/day and 550km = 1 day of rental.
    """
    if not distance_km or distance_km <= 0:
        return None
    days = distance_km / RENTAL_KM_PER_DAY
    return round(days * RENTAL_COST_PER_DAY, 0)

def calculate_reference_sell(pickup_ar, dest_ar, vehicle_ar, lane_data, recommended_sell):
    """
    Calculate reference sell price using cascade:
    1. Recent 90-day median shipper price
    2. Index-adjusted historical median shipper price
    3. Same as recommended sell price
    
    Returns (reference_sell, source)
    """
    # Get recent data
    recent_data = lane_data[lane_data['days_ago'] <= RECENCY_WINDOW] if len(lane_data) > 0 else pd.DataFrame()
    
    # CASCADE 1: Recent median shipper price
    if len(recent_data) > 0 and 'total_shipper_price' in recent_data.columns:
        recent_sell = recent_data['total_shipper_price'].dropna()
        if len(recent_sell) > 0:
            median_sell = recent_sell.median()
            if median_sell > 0:
                return round(median_sell, 0), 'Recent 90d'
    
    # CASCADE 2: Index-adjusted historical median
    if len(lane_data) > 0 and 'total_shipper_price' in lane_data.columns:
        hist_sell = lane_data['total_shipper_price'].dropna()
        if len(hist_sell) > 0:
            hist_median = hist_sell.median()
            if hist_median > 0:
                # Apply index adjustment if we have monthly index
                if 'MONTHLY_INDEX' in config and 'pickup_date' in lane_data.columns:
                    try:
                        # Get average month of historical data
                        lane_data_copy = lane_data.copy()
                        lane_data_copy['pickup_date'] = pd.to_datetime(lane_data_copy['pickup_date'])
                        lane_data_copy['month'] = lane_data_copy['pickup_date'].dt.to_period('M').astype(str)
                        
                        monthly_index = config.get('MONTHLY_INDEX', {})
                        current_index = config.get('CURRENT_INDEX', 1.8)
                        
                        # Calculate average index ratio
                        valid_months = lane_data_copy['month'].value_counts()
                        if len(valid_months) > 0:
                            weighted_index = 0
                            total_weight = 0
                            for month, count in valid_months.items():
                                month_idx = monthly_index.get(month, current_index)
                                if month_idx > 0:
                                    weighted_index += month_idx * count
                                    total_weight += count
                            if total_weight > 0:
                                avg_hist_index = weighted_index / total_weight
                                index_ratio = current_index / avg_hist_index if avg_hist_index > 0 else 1.0
                                adjusted_median = hist_median * index_ratio
                                return round(adjusted_median, 0), 'Historical (idx-adj)'
                    except:
                        pass
                
                # No index adjustment available, return raw historical
                return round(hist_median, 0), 'Historical'
    
    # CASCADE 3: Same as recommended sell
    if recommended_sell:
        return recommended_sell, 'Recommended'
    
    return None, 'N/A'

def get_distance(pickup_ar, dest_ar, lane_data=None):
    """
    Get distance between two cities with consistent priority:
    1. Historical distance from lane data (if provided and has data)
    2. Reverse lane historical distance
    3. DISTANCE_LOOKUP from config
    4. Hardcoded distance matrix (with canonical name fallback)
    5. Reverse in distance matrix
    
    Returns (distance, source)
    """
    lane = f"{pickup_ar} → {dest_ar}"
    reverse_lane = f"{dest_ar} → {pickup_ar}"
    
    # Normalize to canonical names for distance matrix lookup
    # Try exact match first, then lowercase lookup
    pickup_canonical = CITY_TO_CANONICAL.get(pickup_ar)
    if pickup_canonical is None:
        pickup_lower = normalize_english_text(pickup_ar)
        pickup_canonical = CITY_TO_CANONICAL_LOWER.get(pickup_lower, pickup_ar) if pickup_lower else pickup_ar
    
    dest_canonical = CITY_TO_CANONICAL.get(dest_ar)
    if dest_canonical is None:
        dest_lower = normalize_english_text(dest_ar)
        dest_canonical = CITY_TO_CANONICAL_LOWER.get(dest_lower, dest_ar) if dest_lower else dest_ar
    
    # 1. Try historical distance from lane data
    if lane_data is not None and len(lane_data) > 0:
        dist_vals = lane_data[lane_data['distance'] > 0]['distance']
        if len(dist_vals) > 0:
            return dist_vals.median(), 'Historical'
    
    # 2. Try reverse lane in historical data
    reverse_data = df_knn[df_knn['lane'] == reverse_lane]
    if len(reverse_data) > 0:
        dist_vals = reverse_data[reverse_data['distance'] > 0]['distance']
        if len(dist_vals) > 0:
            return dist_vals.median(), 'Historical (reverse)'
    
    # 3. Try DISTANCE_LOOKUP from config (uses lane strings)
    dist = DISTANCE_LOOKUP.get(lane)
    if dist and dist > 0:
        return dist, 'Historical'
    
    # 4. Try reverse in DISTANCE_LOOKUP
    dist = DISTANCE_LOOKUP.get(reverse_lane)
    if dist and dist > 0:
        return dist, 'Historical (reverse)'
    
    # 5. Try distance matrix with original names
    dist = DISTANCE_MATRIX.get((pickup_ar, dest_ar))
    if dist and dist > 0:
        return dist, 'Matrix'
    
    # 6. Try distance matrix with canonical names (key fix for bulk upload)
    if pickup_canonical != pickup_ar or dest_canonical != dest_ar:
        dist = DISTANCE_MATRIX.get((pickup_canonical, dest_canonical))
        if dist and dist > 0:
            return dist, 'Matrix (canonical)'
    
    # 7. Try reverse in distance matrix
    dist = DISTANCE_MATRIX.get((dest_ar, pickup_ar))
    if dist and dist > 0:
        return dist, 'Matrix (reverse)'
    
    # 8. Try reverse with canonical names
    if pickup_canonical != pickup_ar or dest_canonical != dest_ar:
        dist = DISTANCE_MATRIX.get((dest_canonical, pickup_canonical))
        if dist and dist > 0:
            return dist, 'Matrix (canonical reverse)'
    
    # Log missing distance
    log_exception('missing_distance', {
        'pickup_city': pickup_ar,
        'destination_city': dest_ar,
        'pickup_canonical': pickup_canonical,
        'destination_canonical': dest_canonical,
        'pickup_en': to_english_city(pickup_ar),
        'destination_en': to_english_city(dest_ar),
    })
    
    return 0, 'Missing'

# ============================================
# ROUNDING HELPERS
# ============================================
def round_to_nearest(value, nearest):
    """Round to nearest value (e.g., 100 or 50)."""
    if value is None or pd.isna(value):
        return None
    return int(round(value / nearest) * nearest)

# ============================================
# PRICE CASCADE FUNCTION
# ============================================
def calculate_prices(pickup_ar, dest_ar, vehicle_ar, distance_km, lane_data=None):
    """
    Calculate Buy Price and Sell Price using cascade:
    1. Recency (median of recent loads)
    2. Index + Shrinkage (actual prediction)
    3. Blend 0.7 (actual prediction)
    
    Returns dict with pricing info including reference sell and rental cost.
    """
    lane = f"{pickup_ar} → {dest_ar}"
    
    # Get lane data if not provided
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
    
    # CASCADE 1: Recency (median of recent loads)
    if recent_count >= 1:
        recent_median = recent_data['total_carrier_price'].median()
        buy_price = recent_median
        model_used = 'Recency'
        confidence = 'High' if recent_count >= 5 else 'Medium' if recent_count >= 2 else 'Low'
    
    # CASCADE 2: Index + Shrinkage (actual prediction, not upper bound)
    elif index_shrink_predictor and index_shrink_predictor.has_lane_data(lane):
        pred = index_shrink_predictor.predict(pickup_ar, dest_ar, distance_km)
        buy_price = pred.get('predicted_cost')  # Actual prediction
        model_used = pred['method']
        confidence = pred['confidence']
    
    # CASCADE 3: Blend 0.7 (actual prediction, not upper bound)
    elif blend_predictor:
        pred = blend_predictor.predict(pickup_ar, dest_ar, distance_km)
        buy_price = pred.get('predicted_cost')  # Actual prediction
        model_used = pred['method']
        confidence = pred['confidence']
    
    # Fallback
    else:
        if distance_km and distance_km > 0:
            buy_price = distance_km * 1.8  # Default CPK (actual, not inflated)
            model_used = 'Default CPK'
            confidence = 'Very Low'
    
    # Round buy price to nearest 100
    buy_price_rounded = round_to_nearest(buy_price, 100)
    
    # Calculate recommended sell price based on backhaul probability
    backhaul_prob, margin, backhaul_ratio = get_backhaul_probability(dest_ar)
    
    if buy_price_rounded:
        sell_price = buy_price_rounded * (1 + margin)
        sell_price_rounded = round_to_nearest(sell_price, 50)
    else:
        sell_price_rounded = None
    
    # Calculate reference sell price (from actual market data)
    ref_sell, ref_sell_source = calculate_reference_sell(
        pickup_ar, dest_ar, vehicle_ar, lane_data, sell_price_rounded
    )
    ref_sell_rounded = round_to_nearest(ref_sell, 50) if ref_sell else sell_price_rounded
    
    # Calculate rental cost index
    rental_cost = calculate_rental_cost(distance_km)
    
    return {
        'buy_price': buy_price_rounded,
        'sell_price': sell_price_rounded,  # Recommended sell (buy + margin)
        'ref_sell_price': ref_sell_rounded,  # Reference sell (from market data)
        'ref_sell_source': ref_sell_source,
        'rental_cost': rental_cost,
        'target_margin': f"{margin:.0%}",
        'backhaul_probability': backhaul_prob,
        'backhaul_ratio': round(backhaul_ratio, 2) if backhaul_ratio else None,
        'model_used': model_used,
        'confidence': confidence,
        'recent_count': recent_count,
    }

# ============================================
# SAMPLE SELECTION FOR AMMUNITION
# ============================================
def select_spaced_samples(df, n_samples, min_days_apart=MIN_DAYS_APART):
    if len(df) == 0:
        return pd.DataFrame()
    df = df.sort_values('days_ago', ascending=True)
    selected = []
    last_days_ago = -min_days_apart
    recent_df = df[df['days_ago'] <= RECENT_PRIORITY_DAYS]
    for _, row in recent_df.iterrows():
        if row['days_ago'] >= last_days_ago + min_days_apart:
            selected.append(row)
            last_days_ago = row['days_ago']
            if len(selected) >= n_samples:
                break
    if len(selected) < n_samples:
        older_df = df[df['days_ago'] > RECENT_PRIORITY_DAYS]
        for _, row in older_df.iterrows():
            if row['days_ago'] >= last_days_ago + min_days_apart:
                selected.append(row)
                last_days_ago = row['days_ago']
                if len(selected) >= n_samples:
                    break
    return pd.DataFrame(selected) if selected else pd.DataFrame()

def get_ammunition_loads(lane, vehicle_ar, commodity=None):
    matches = df_knn[
        (df_knn['lane'] == lane) & 
        (df_knn['vehicle_type'] == vehicle_ar) &
        (df_knn['days_ago'] <= MAX_AGE_DAYS)
    ].copy()
    
    if len(matches) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    if commodity and commodity not in ['', 'Auto', 'auto', None]:
        same_commodity = matches[matches['commodity'] == commodity]
        other_commodity = matches[matches['commodity'] != commodity]
    else:
        if len(matches) > 0:
            most_common = matches['commodity'].mode().iloc[0]
            same_commodity = matches[matches['commodity'] == most_common]
            other_commodity = matches[matches['commodity'] != most_common]
        else:
            same_commodity = pd.DataFrame()
            other_commodity = pd.DataFrame()
    
    return select_spaced_samples(same_commodity, 5), select_spaced_samples(other_commodity, 5)

# ============================================
# BULK LOOKUP
# ============================================
def lookup_route_stats(pickup_ar, dest_ar, vehicle_ar=None):
    """Lookup route with Buy/Sell pricing for bulk output."""
    if vehicle_ar is None or vehicle_ar in ['', 'Auto', 'auto', None]:
        vehicle_ar = DEFAULT_VEHICLE_AR
    
    lane = f"{pickup_ar} → {dest_ar}"
    
    # Get lane data
    lane_data = df_knn[
        (df_knn['lane'] == lane) & 
        (df_knn['vehicle_type'] == vehicle_ar)
    ].copy()
    
    # Get distance
    distance, distance_source = get_distance(pickup_ar, dest_ar, lane_data)
    
    # Calculate prices using cascade
    pricing = calculate_prices(pickup_ar, dest_ar, vehicle_ar, distance, lane_data)
    
    # Clean bulk output with clear column names
    result = {
        'From': to_english_city(pickup_ar),
        'To': to_english_city(dest_ar),
        'Distance': int(distance) if distance else 0,
        'Buy_Price': pricing['buy_price'],
        'Rec_Sell': pricing['sell_price'],  # Recommended sell (buy + margin)
        'Ref_Sell': pricing['ref_sell_price'],  # Reference sell (market data)
        'Ref_Sell_Src': pricing['ref_sell_source'],
        'Rental_Cost': pricing['rental_cost'],
        'Margin': pricing['target_margin'],
        'Model': pricing['model_used'],
        'Confidence': pricing['confidence'],
        'Recent_N': pricing['recent_count'],
    }
    
    return result

# ============================================
# SINGLE ROUTE PRICING
# ============================================
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
    
    hist_count = len(lane_data) if len(lane_data) > 0 else 0
    hist_min = lane_data['total_carrier_price'].min() if hist_count > 0 else None
    hist_max = lane_data['total_carrier_price'].max() if hist_count > 0 else None
    hist_median = lane_data['total_carrier_price'].median() if hist_count > 0 else None
    
    recent_count = len(recent_data) if len(recent_data) > 0 else 0
    recent_min = recent_data['total_carrier_price'].min() if recent_count > 0 else None
    recent_max = recent_data['total_carrier_price'].max() if recent_count > 0 else None
    recent_median = recent_data['total_carrier_price'].median() if recent_count > 0 else None
    
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
    
    if len(lane_data) > 0:
        container = int(lane_data['container'].mode().iloc[0])
    else:
        container = 0
    
    is_same_city = (pickup_ar == dest_ar)
    is_multistop = 0 if is_same_city else (int(lane_data['is_multistop'].mode().iloc[0]) if len(lane_data) > 0 else 0)
    
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

# ============================================
# BUILD DROPDOWN OPTIONS
# ============================================
pickup_cities_ar = sorted(df_knn['pickup_city'].unique())
pickup_cities_en = sorted(set([to_english_city(c) for c in pickup_cities_ar]))
dest_cities_ar = sorted(df_knn['destination_city'].unique())
dest_cities_en = sorted(set([to_english_city(c) for c in dest_cities_ar]))
vehicle_types_ar = df_knn['vehicle_type'].unique()
vehicle_types_en = sorted(set([to_english_vehicle(v) for v in vehicle_types_ar]))
commodities_ar = sorted(df_knn['commodity'].unique())
commodities = sorted(set([to_english_commodity(c) for c in commodities_ar]))

# ============================================
# APP UI
# ============================================
st.title("🚚 Freight Pricing Tool")
model_status = []
if index_shrink_predictor:
    model_status.append("✅ Index+Shrinkage")
if blend_predictor:
    model_status.append("✅ Blend 0.7")
dist_status = f"✅ {len(DISTANCE_MATRIX):,} distances" if DISTANCE_MATRIX else ""
st.caption(f"ML-powered pricing | Domestic | {' | '.join(model_status)} | {dist_status}")

tab1, tab2 = st.tabs(["🎯 Single Route Pricing", "📦 Bulk Route Lookup"])

# ============================================
# TAB 1: SINGLE ROUTE PRICING
# ============================================
with tab1:
    st.subheader("📋 Route Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        pickup_en = st.selectbox("Pickup City", options=pickup_cities_en,
            index=pickup_cities_en.index('Jeddah') if 'Jeddah' in pickup_cities_en else 0, key='single_pickup')
        pickup_city = to_arabic_city(pickup_en)

    with col2:
        dest_en = st.selectbox("Destination City", options=dest_cities_en,
            index=dest_cities_en.index('Riyadh') if 'Riyadh' in dest_cities_en else 0, key='single_dest')
        destination_city = to_arabic_city(dest_en)

    with col3:
        default_idx = vehicle_types_en.index(DEFAULT_VEHICLE_EN) if DEFAULT_VEHICLE_EN in vehicle_types_en else 0
        vehicle_en = st.selectbox("Vehicle Type", options=vehicle_types_en, index=default_idx, key='single_vehicle')
        vehicle_type = to_arabic_vehicle(vehicle_en)

    st.subheader("📦 Optional Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        container_options = ['Auto-detect', 'Yes (Container)', 'No (Non-container)']
        st.selectbox("Container", options=container_options, key='single_container')

    with col2:
        commodity_options = ['Auto-detect'] + commodities
        commodity_select = st.selectbox("Commodity", options=commodity_options, key='single_commodity')
        commodity_input = None if commodity_select == 'Auto-detect' else to_arabic_commodity(commodity_select)

    with col3:
        weight = st.number_input("Weight (Tons)", min_value=0.0, max_value=100.0, value=0.0, step=1.0,
                                 help="Leave as 0 for auto-detect", key='single_weight')
        weight = None if weight == 0 else weight

    st.markdown("---")
    if st.button("🎯 Generate Pricing", type="primary", use_container_width=True, key='single_generate'):
        result = price_single_route(pickup_city, destination_city, vehicle_type, commodity_input, weight)
        lane_en = f"{pickup_en} → {dest_en}"
        lane_ar = f"{pickup_city} → {destination_city}"
        
        # Store in session for report button
        st.session_state.last_result = result
        st.session_state.last_lane = {'pickup_ar': pickup_city, 'dest_ar': destination_city, 
                                       'pickup_en': pickup_en, 'dest_en': dest_en}
        
        st.markdown("---")
        
        # Check distance
        if result['Distance_km'] == 0 or result['Distance_Source'] == 'Default':
            st.error(f"⚠️ Distance data missing or estimated for this route ({result['Distance_km']} km)")
        
        is_rare_lane = result.get('Is_Rare_Lane', False)
        
        # MAIN PRICING DISPLAY
        st.header("💰 Pricing Results")
        st.info(f"**{lane_en}** | 🚛 {result['Vehicle_Type']} | 📏 {result['Distance_km']:.0f} km | ⚖️ {result['Weight_Tons']:.1f} T")
        
        # Primary pricing row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🛒 BUY PRICE", f"{result['Buy_Price']:,} SAR" if result['Buy_Price'] else "N/A")
        with col2:
            st.metric("💵 REC. SELL", f"{result['Rec_Sell_Price']:,} SAR" if result.get('Rec_Sell_Price') else "N/A",
                     help="Recommended sell = Buy + Target Margin")
        with col3:
            ref_sell = result.get('Ref_Sell_Price')
            ref_src = result.get('Ref_Sell_Source', 'N/A')
            st.metric("📈 REF. SELL", f"{ref_sell:,} SAR" if ref_sell else "N/A",
                     help=f"Reference from market data ({ref_src})")
        
        # Secondary info row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Margin", result['Target_Margin'])
        with col2:
            backhaul_emoji = {'High': '🟢', 'Medium': '🟡', 'Low': '🔴', 'Unknown': '⚪'}.get(result['Backhaul_Probability'], '⚪')
            st.metric("🔄 Backhaul", f"{backhaul_emoji} {result['Backhaul_Probability']}")
        with col3:
            rental = result.get('Rental_Cost')
            st.metric("🚛 Rental Index", f"{rental:,.0f} SAR" if rental else "N/A",
                     help="Based on 800 SAR/day, 550 km/day")
        with col4:
            conf_emoji = {'High': '🟢', 'Medium': '🟡', 'Low': '🟠', 'Very Low': '🔴'}.get(result['Confidence'], '⚪')
            st.metric("📊 Confidence", f"{conf_emoji} {result['Confidence']}")
        
        # Model info
        st.caption(f"📊 Model: **{result['Model_Used']}** | CPK: {result['Cost_Per_KM']} SAR/km | Ref. Sell Source: {result.get('Ref_Sell_Source', 'N/A')}")
        
        if is_rare_lane:
            st.warning(f"⚠️ **Rare Lane** - No loads in last {RECENCY_WINDOW} days. Using model prediction.")
        
        # Historical data
        st.markdown("---")
        st.subheader(f"📊 Price History")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Historical (All Time)**")
            if result['Hist_Count'] > 0:
                st.dataframe(pd.DataFrame({
                    'Metric': ['Count', 'Min', 'Median', 'Max'],
                    'Value': [f"{result['Hist_Count']} loads", f"{result['Hist_Min']:,.0f} SAR",
                              f"{result['Hist_Median']:,.0f} SAR", f"{result['Hist_Max']:,.0f} SAR"]
                }), use_container_width=True, hide_index=True)
            else:
                st.warning("No historical data")
        
        with col2:
            st.markdown(f"**Recent ({RECENCY_WINDOW} Days)**")
            if result[f'Recent_{RECENCY_WINDOW}d_Count'] > 0:
                st.dataframe(pd.DataFrame({
                    'Metric': ['Count', 'Min', 'Median', 'Max'],
                    'Value': [f"{result[f'Recent_{RECENCY_WINDOW}d_Count']} loads",
                              f"{result[f'Recent_{RECENCY_WINDOW}d_Min']:,.0f} SAR",
                              f"{result[f'Recent_{RECENCY_WINDOW}d_Median']:,.0f} SAR",
                              f"{result[f'Recent_{RECENCY_WINDOW}d_Max']:,.0f} SAR"]
                }), use_container_width=True, hide_index=True)
            else:
                st.warning(f"No loads in last {RECENCY_WINDOW} days")
        
        # Model Comparison
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
        
        # AMMUNITION
        if not is_rare_lane:
            st.markdown("---")
            st.subheader("🚚 Your Ammunition (Recent Matches)")
            
            same_samples, other_samples = get_ammunition_loads(lane_ar, vehicle_type, commodity_input)
            
            commodity_used = to_english_commodity(commodity_input) if commodity_input else result['Commodity']
            if len(same_samples) > 0:
                st.markdown(f"**Same Commodity ({commodity_used}):**")
                same_samples['Lane_EN'] = same_samples['pickup_city'].apply(to_english_city) + ' → ' + same_samples['destination_city'].apply(to_english_city)
                same_samples['Commodity_EN'] = same_samples['commodity'].apply(to_english_commodity)
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
    
    # REPORT BUTTON - Outside generate block
    if 'last_result' in st.session_state and 'last_lane' in st.session_state:
        st.markdown("---")
        with st.expander("🚨 Report Issue with Distance/Data", expanded=False):
            st.caption(f"Reporting for: {st.session_state.last_lane['pickup_en']} → {st.session_state.last_lane['dest_en']}")
            
            issue_type = st.selectbox("Issue Type", 
                ["Distance is incorrect", "Distance is 0/missing", "Data looks wrong", "Other"],
                key='report_issue_type')
            user_notes = st.text_area("Additional Notes (optional)", key='report_notes')
            
            if st.button("📤 Submit Report", key='submit_report'):
                lane_info = st.session_state.last_lane
                result_info = st.session_state.last_result
                success = report_distance_issue(
                    lane_info['pickup_ar'], 
                    lane_info['dest_ar'],
                    lane_info['pickup_en'], 
                    lane_info['dest_en'],
                    result_info['Distance_km'],
                    issue_type,
                    user_notes
                )
                if success:
                    st.success("✅ Report submitted! Thank you.")
                else:
                    st.warning("Could not submit report. Google Sheets not configured.")

# ============================================
# TAB 2: BULK ROUTE LOOKUP
# ============================================
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
    
    sample_df = pd.DataFrame({
        'From': ['Jeddah', 'Jeddah', 'Riyadh'],
        'To': ['Riyadh', 'Dammam', 'Jeddah'],
        'Vehicle_Type': ['Flatbed Trailer', '', ''],
    })
    
    st.download_button("📥 Download Template", sample_df.to_csv(index=False), "lookup_template.csv", "text/csv")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            routes_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(routes_df)} routes")
            
            with st.expander("📋 Preview"):
                st.dataframe(routes_df.head(10), use_container_width=True)
            
            if st.button("🔍 Look Up All Routes", type="primary", use_container_width=True):
                results = []
                unmatched_cities = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in routes_df.iterrows():
                    pickup_raw = str(row.get('From', '')).strip()
                    dest_raw = str(row.get('To', '')).strip()
                    vehicle_raw = str(row.get('Vehicle_Type', '')).strip() if 'Vehicle_Type' in row else ''
                    
                    pickup_ar, pickup_matched = normalize_city(pickup_raw)
                    dest_ar, dest_matched = normalize_city(dest_raw)
                    
                    if not pickup_matched and pickup_ar not in VALID_CITIES_AR:
                        unmatched_cities.append({'Row': idx+1, 'Column': 'From', 'Original': pickup_raw})
                        log_exception('unmatched_city', {
                            'row': idx+1,
                            'column': 'From',
                            'original_value': pickup_raw,
                            'normalized_to': pickup_ar,
                        })
                    if not dest_matched and dest_ar not in VALID_CITIES_AR:
                        unmatched_cities.append({'Row': idx+1, 'Column': 'To', 'Original': dest_raw})
                        log_exception('unmatched_city', {
                            'row': idx+1,
                            'column': 'To',
                            'original_value': dest_raw,
                            'normalized_to': dest_ar,
                        })
                    
                    if vehicle_raw in ['', 'nan', 'None', 'Auto', 'auto']:
                        vehicle_ar = DEFAULT_VEHICLE_AR
                    else:
                        vehicle_ar = to_arabic_vehicle(vehicle_raw)
                    
                    status_text.text(f"Looking up {idx+1}/{len(routes_df)}: {pickup_raw} → {dest_raw}")
                    
                    result = lookup_route_stats(pickup_ar, dest_ar, vehicle_ar)
                    results.append(result)
                    progress_bar.progress((idx + 1) / len(routes_df))
                
                status_text.text("✅ Complete!")
                results_df = pd.DataFrame(results)
                
                st.markdown("---")
                st.subheader("📊 Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Routes", len(results_df))
                with col2:
                    recency_count = (results_df['Model'] == 'Recency').sum()
                    st.metric("Recent Data", recency_count)
                with col3:
                    model_count = results_df['Model'].str.contains('Index|Shrink|Blend', na=False).sum()
                    st.metric("Model Est.", model_count)
                with col4:
                    high_conf = (results_df['Confidence'] == 'High').sum()
                    st.metric("High Conf.", high_conf)
                
                # Column explanations
                st.caption("""
                **Columns:** Buy_Price = Carrier cost | Rec_Sell = Buy + Margin | Ref_Sell = Market reference (source in Ref_Sell_Src) | Rental_Cost = 800 SAR/day × (km ÷ 550)
                """)
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                st.download_button("📥 Download Results", results_df.to_csv(index=False), 
                                  "route_lookup_results.csv", "text/csv", type="primary")
                
                if len(unmatched_cities) > 0:
                    st.markdown("---")
                    st.subheader("⚠️ Unmatched Cities")
                    st.warning(f"{len(unmatched_cities)} cities could not be matched.")
                    st.dataframe(pd.DataFrame(unmatched_cities), use_container_width=True, hide_index=True)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================
# ERROR LOG SECTION
# ============================================
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
