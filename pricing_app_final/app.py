import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import pickle
import os
import io
import re

st.set_page_config(page_title="Freight Pricing Tool", page_icon="🚚", layout="wide")

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================
# FIXED SETTINGS
# ============================================
ANCHOR_DISCOUNT = 12
CEILING_PREMIUM = 5
N_SIMILAR = 10
MIN_DAYS_APART = 5
MAX_AGE_DAYS = 180
RECENT_PRIORITY_DAYS = 30
RECENCY_WINDOW = 90

# Error bar percentages by confidence level
ERROR_BARS = {
    'High': 0.10,      # ±10%
    'Medium': 0.15,    # ±15%
    'Low': 0.25,       # ±25%
    'Very Low': 0.35   # ±35%
}

# ============================================
# TRANSLATION MAPPINGS
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
    'Khamis Mushait': 'Khamis Mushait',
    'Abha': 'Abha',
    'Umluj': 'Umluj',
    'الجبيل': 'Jubail',
}

CITY_AR = {v: k for k, v in CITY_EN.items()}

CITY_MAPPING_EN_TO_AR = {
    'Jeddah': 'جدة', 'Jiddah': 'جدة', 'Jedda': 'جدة',
    'Riyadh': 'الرياض', 'Riyad': 'الرياض',
    'Dammam': 'الدمام', 'Dammam ': 'الدمام', 'Damam': 'الدمام',
    'Makkah': 'مكة المكرمة', 'Mecca': 'مكة المكرمة', 'Mekkah': 'مكة المكرمة',
    'Madina': 'المدينة المنورة', 'Madinah': 'المدينة المنورة', 'Medina': 'المدينة المنورة',
    'Rabigh': 'رابغ', 'Yanbu': 'ينبع', 'Yenbu': 'ينبع',
    'Tabuk': 'تبوك', 'Tabouk': 'تبوك',
    'Taif': 'الطائف', 'Tayef': 'الطائف',
    'Al Hasa': 'الْأَحْسَاء', 'Al-Hasa': 'الْأَحْسَاء', 'Ahsa': 'الْأَحْسَاء',
    'Al Kharj': 'الخرج', 'Al-Kharij': 'الخرج', 'Kharj': 'الخرج',
    'Al Qassim': 'القصيم', 'Qassim': 'القصيم', 'Qaseem': 'القصيم',
    'Al Baha': 'ٱلْبَاحَة', 'Baha': 'ٱلْبَاحَة',
    'Jizan': 'جازان', 'Jazan': 'جازان', 'Gizan': 'جازان',
    'Najran': 'نجران', 'Nejran': 'نجران',
    'Abha': 'Abha', 'Abaha': 'Abha',
    'Khamis Mushait': 'Khamis Mushait', 'Khamis': 'Khamis Mushait',
    'Arar': 'عرعر', 'Skaka': 'سكاكا', 'Sakaka': 'سكاكا',
    'Hafar Al Batin': 'حفر الباطن', 'Hafr Al Batin': 'حفر الباطن',
    'Neom': 'تبوك', 'NEOM': 'تبوك',
    'Sudair': 'سدير', 'Umluj': 'Umluj',
    'Jubail': 'الجبيل', 'Al Jubail': 'الجبيل',
}

CITY_MAPPING_AR_TO_AR = {
    'جدة': 'جدة', 'جده': 'جدة',
    'الرياض': 'الرياض', 'رياض': 'الرياض',
    'الدمام': 'الدمام', 'دمام': 'الدمام',
    'مكة': 'مكة المكرمة', 'مكه': 'مكة المكرمة', 'مكة المكرمة': 'مكة المكرمة',
    'المدينة': 'المدينة المنورة', 'المدينه': 'المدينة المنورة', 'المدينة المنورة': 'المدينة المنورة',
    'الطائف': 'الطائف', 'طائف': 'الطائف',
    'تبوك': 'تبوك', 'القصيم': 'القصيم', 'قصيم': 'القصيم',
    'الخرج': 'الخرج', 'خرج': 'الخرج',
    'جازان': 'جازان', 'جيزان': 'جازان',
    'نجران': 'نجران', 'عرعر': 'عرعر', 'سكاكا': 'سكاكا',
    'ينبع': 'ينبع', 'رابغ': 'رابغ',
    'الاحساء': 'الْأَحْسَاء', 'الأحساء': 'الْأَحْسَاء', 'الْأَحْسَاء': 'الْأَحْسَاء',
    'الباحة': 'ٱلْبَاحَة', 'ٱلْبَاحَة': 'ٱلْبَاحَة',
    'حفر الباطن': 'حفر الباطن', 'سدير': 'سدير',
    'ابها': 'Abha', 'أبها': 'Abha',
    'خميس مشيط': 'Khamis Mushait', 'الجبيل': 'الجبيل',
}

def normalize_city(city_raw):
    if pd.isna(city_raw) or city_raw == '':
        return None, False
    city = str(city_raw).strip()
    city = re.sub(r'\s+', ' ', city).strip()
    is_arabic = bool(re.search(r'[\u0600-\u06FF]', city))
    if is_arabic:
        if city in CITY_MAPPING_AR_TO_AR:
            return CITY_MAPPING_AR_TO_AR[city], True
        for ar_variant, standard in CITY_MAPPING_AR_TO_AR.items():
            if ar_variant in city or city in ar_variant:
                return standard, True
        return city, False
    else:
        city_lower = city.lower()
        for en_variant, ar_standard in CITY_MAPPING_EN_TO_AR.items():
            if en_variant.lower() == city_lower:
                return ar_standard, True
        for en_variant, ar_standard in CITY_MAPPING_EN_TO_AR.items():
            if en_variant.lower() in city_lower or city_lower in en_variant.lower():
                return ar_standard, True
        return city, False

def to_english_city(city_ar):
    return CITY_EN.get(city_ar, city_ar)

def to_arabic_city(city_en):
    if city_en in CITY_AR:
        return CITY_AR[city_en]
    normalized, _ = normalize_city(city_en)
    return normalized if normalized else city_en

def to_english_vehicle(vtype_ar):
    return VEHICLE_TYPE_EN.get(vtype_ar, vtype_ar)

def to_arabic_vehicle(vtype_en):
    if vtype_en in VEHICLE_TYPE_AR:
        return VEHICLE_TYPE_AR[vtype_en]
    if vtype_en in VEHICLE_TYPE_EN:
        return vtype_en
    return DEFAULT_VEHICLE_AR

# ============================================
# RARE LANE PREDICTOR CLASS
# ============================================
class RareLanePredictor:
    def __init__(self, model_artifacts):
        m = model_artifacts
        self.current_index = m['current_index']
        self.lane_multipliers = m['lane_multipliers']
        self.global_mean = m['global_mean']
        self.k = m['k_prior_strength']
        self.pickup_priors = m['pickup_priors']
        self.dest_priors = m['dest_priors']
        self.lane_stats = m['lane_stats']
        self.city_to_region = m['city_to_region']
        self.regional_cpk = {tuple(k.split('|')): v for k, v in m['regional_cpk'].items()}
        self.model_date = m['model_date']
    
    def _predict_index(self, lane):
        if lane in self.lane_multipliers:
            return self.current_index * self.lane_multipliers[lane]
        return None
    
    def _predict_shrinkage(self, lane, pickup_city, dest_city):
        p_prior = self.pickup_priors.get(pickup_city, self.global_mean)
        d_prior = self.dest_priors.get(dest_city, self.global_mean)
        city_prior = (p_prior + d_prior) / 2
        
        if lane in self.lane_stats:
            stats = self.lane_stats[lane]
            lane_mean = stats['lane_mean']
            lane_n = stats['lane_n']
            lam = lane_n / (lane_n + self.k)
            return lam * lane_mean + (1 - lam) * city_prior
        else:
            return city_prior
    
    def _predict_regional(self, pickup_city, dest_city):
        p_region = self.city_to_region.get(pickup_city)
        d_region = self.city_to_region.get(dest_city)
        if p_region and d_region:
            return self.regional_cpk.get((p_region, d_region))
        return None
    
    def predict(self, pickup_city, dest_city, distance_km=None):
        lane = f"{pickup_city} → {dest_city}"
        
        idx_pred = self._predict_index(lane)
        shrink_pred = self._predict_shrinkage(lane, pickup_city, dest_city)
        regional_pred = self._predict_regional(pickup_city, dest_city)
        
        if idx_pred is not None and shrink_pred is not None:
            predicted_cpk = (idx_pred + shrink_pred) / 2
            method = 'Index + Shrink Ensemble'
            if lane in self.lane_stats:
                n = self.lane_stats[lane]['lane_n']
                confidence = 'High' if n >= 20 else 'Medium' if n >= 5 else 'Low'
            else:
                confidence = 'Low'
        elif shrink_pred is not None:
            predicted_cpk = shrink_pred
            method = 'Shrinkage (city priors)'
            confidence = 'Medium'
        elif regional_pred is not None:
            predicted_cpk = regional_pred
            method = 'Regional Fallback'
            confidence = 'Low'
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
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    MODEL_DIR = os.path.join(APP_DIR, 'model_export')
    
    carrier_model = CatBoostRegressor()
    shipper_model = CatBoostRegressor()
    
    json_path = os.path.join(MODEL_DIR, 'carrier_model.json')
    if os.path.exists(json_path):
        carrier_model.load_model(os.path.join(MODEL_DIR, 'carrier_model.json'), format='json')
        shipper_model.load_model(os.path.join(MODEL_DIR, 'shipper_model.json'), format='json')
    else:
        carrier_model.load_model(os.path.join(MODEL_DIR, 'carrier_model.cbm'))
        shipper_model.load_model(os.path.join(MODEL_DIR, 'shipper_model.cbm'))
    
    with open(os.path.join(MODEL_DIR, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)
    
    csv_path = os.path.join(MODEL_DIR, 'reference_data.csv')
    parquet_path = os.path.join(MODEL_DIR, 'reference_data.parquet')
    if os.path.exists(csv_path):
        df_knn = pd.read_csv(csv_path)
    else:
        df_knn = pd.read_parquet(parquet_path)
    
    # Load rare lane model if available
    rare_lane_path = os.path.join(MODEL_DIR, 'rare_lane_models.pkl')
    rare_lane_predictor = None
    if os.path.exists(rare_lane_path):
        with open(rare_lane_path, 'rb') as f:
            rare_lane_artifacts = pickle.load(f)
        rare_lane_predictor = RareLanePredictor(rare_lane_artifacts)
    
    return {
        'carrier_model': carrier_model,
        'shipper_model': shipper_model,
        'config': config,
        'df_knn': df_knn,
        'rare_lane_predictor': rare_lane_predictor
    }

models = load_models()
config = models['config']
df_knn = models['df_knn']
rare_lane_predictor = models['rare_lane_predictor']

FEATURES = config['FEATURES']
ENTITY_MAPPING = config.get('ENTITY_MAPPING', 'Domestic')
DISTANCE_LOOKUP = config.get('DISTANCE_LOOKUP', {})

df_knn = df_knn[df_knn['entity_mapping'] == ENTITY_MAPPING].copy()
VALID_CITIES_AR = set(df_knn['pickup_city'].unique()) | set(df_knn['destination_city'].unique())

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
# BULK LOOKUP WITH RARE LANE
# ============================================
def lookup_route_stats(pickup_ar, dest_ar, vehicle_ar=None):
    if vehicle_ar is None or vehicle_ar in ['', 'Auto', 'auto', None]:
        vehicle_ar = DEFAULT_VEHICLE_AR
    
    lane = f"{pickup_ar} → {dest_ar}"
    
    lane_data = df_knn[
        (df_knn['lane'] == lane) & 
        (df_knn['vehicle_type'] == vehicle_ar)
    ].copy()
    
    recent_data = lane_data[lane_data['days_ago'] <= RECENCY_WINDOW]
    
    # Historical stats
    if len(lane_data) > 0:
        hist_count = len(lane_data)
        hist_min = int(lane_data['total_carrier_price'].min())
        hist_max = int(lane_data['total_carrier_price'].max())
        hist_median = int(lane_data['total_carrier_price'].median())
        distance = lane_data['distance'].median()
    else:
        hist_count = 0
        hist_min = hist_max = hist_median = None
        distance = DISTANCE_LOOKUP.get(lane, 500)
    
    # Recent stats
    if len(recent_data) > 0:
        recent_count = len(recent_data)
        recent_min = int(recent_data['total_carrier_price'].min())
        recent_max = int(recent_data['total_carrier_price'].max())
        recent_median = int(recent_data['total_carrier_price'].median())
    else:
        recent_count = 0
        recent_min = recent_max = recent_median = None
    
    result = {
        'From': to_english_city(pickup_ar),
        'To': to_english_city(dest_ar),
        'Vehicle_Type': to_english_vehicle(vehicle_ar),
        'Distance_km': int(distance),
        'Hist_Count': hist_count,
        'Hist_Min': hist_min,
        'Hist_Median': hist_median,
        'Hist_Max': hist_max,
        f'Recent_{RECENCY_WINDOW}d_Count': recent_count,
        f'Recent_{RECENCY_WINDOW}d_Min': recent_min,
        f'Recent_{RECENCY_WINDOW}d_Median': recent_median,
        f'Recent_{RECENCY_WINDOW}d_Max': recent_max,
    }
    
    # Add rare lane prediction if available and useful
    if rare_lane_predictor:
        rare_pred = rare_lane_predictor.predict(pickup_ar, dest_ar, distance)
        result['Model_CPK'] = rare_pred['predicted_cpk']
        result['Model_Price_Low'] = rare_pred.get('cost_low')
        result['Model_Price'] = rare_pred.get('predicted_cost')
        result['Model_Price_High'] = rare_pred.get('cost_high')
        result['Model_Method'] = rare_pred['method']
        result['Model_Confidence'] = rare_pred['confidence']
    
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
    
    is_same_city = (pickup_ar == dest_ar)
    if is_same_city and len(lane_data) > 0:
        local_data = lane_data[lane_data['is_multistop'] == 0]
        local_recent = local_data[local_data['days_ago'] <= RECENCY_WINDOW]
        if len(local_data) > 0:
            lane_data = local_data
            recent_data = local_recent
    
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
    
    if len(lane_data) > 0:
        container = int(lane_data['container'].mode().iloc[0])
        distance = lane_data['distance'].median()
    else:
        container = 0
        distance = DISTANCE_LOOKUP.get(lane, 500)
    
    is_multistop = 0 if is_same_city else (int(lane_data['is_multistop'].mode().iloc[0]) if len(lane_data) > 0 else 0)
    
    input_dict = {'entity_mapping': ENTITY_MAPPING, 'pickup_city': pickup_ar, 'destination_city': dest_ar}
    if 'commodity' in FEATURES: input_dict['commodity'] = commodity
    if 'vehicle_type' in FEATURES: input_dict['vehicle_type'] = vehicle_ar
    if 'distance' in FEATURES: input_dict['distance'] = distance
    if 'weight' in FEATURES: input_dict['weight'] = weight
    if 'container' in FEATURES: input_dict['container'] = container
    if 'is_multistop' in FEATURES: input_dict['is_multistop'] = is_multistop
    
    input_data = pd.DataFrame([input_dict])
    pred_carrier = models['carrier_model'].predict(input_data[FEATURES])[0]
    pred_shipper = models['shipper_model'].predict(input_data[FEATURES])[0]
    
    if recent_count >= 2:
        actual_avg = recent_data['total_carrier_price'].mean()
        divergence = abs(pred_carrier - actual_avg) / actual_avg
        if divergence > 0.3:
            recommended = 0.8 * actual_avg + 0.2 * pred_carrier
            source = "80% recent + 20% model"
        else:
            recommended = 0.5 * actual_avg + 0.5 * pred_carrier
            source = "50/50 blend"
    elif hist_count >= 2:
        actual_avg = lane_data['total_carrier_price'].mean()
        recommended = 0.6 * actual_avg + 0.4 * pred_carrier
        source = "60% hist + 40% model"
    else:
        recommended = pred_carrier
        source = "Model only"
    
    anchor = recommended * (1 - ANCHOR_DISCOUNT / 100)
    target = recommended
    ceiling = recommended * (1 + CEILING_PREMIUM / 100)
    cost_per_km = recommended / distance if distance > 0 else None
    
    result = {
        'Vehicle_Type': to_english_vehicle(vehicle_ar),
        'Commodity': commodity,
        'Weight_Tons': round(weight, 1),
        'Distance_km': round(distance, 0),
        'Hist_Count': hist_count,
        'Hist_Min': round(hist_min, 0) if hist_min else None,
        'Hist_Median': round(hist_median, 0) if hist_median else None,
        'Hist_Max': round(hist_max, 0) if hist_max else None,
        f'Recent_{RECENCY_WINDOW}d_Count': recent_count,
        f'Recent_{RECENCY_WINDOW}d_Min': round(recent_min, 0) if recent_min else None,
        f'Recent_{RECENCY_WINDOW}d_Median': round(recent_median, 0) if recent_min else None,
        f'Recent_{RECENCY_WINDOW}d_Max': round(recent_max, 0) if recent_min else None,
        'Recommended_Carrier': round(recommended, 0),
        'Recommendation_Source': source,
        'Shipper_Rate': round(pred_shipper, 0),
        'Cost_Per_KM': round(cost_per_km, 2) if cost_per_km else None,
        'Anchor': round(anchor, 0),
        'Target': round(target, 0),
        'Ceiling': round(ceiling, 0),
        'Margin': round(pred_shipper - recommended, 0),
        'Margin_Pct': round((pred_shipper - recommended) / pred_shipper * 100, 1) if pred_shipper > 0 else None,
    }
    
    # Add rare lane prediction for comparison
    if rare_lane_predictor:
        rare_pred = rare_lane_predictor.predict(pickup_ar, dest_ar, distance)
        result['RareLane_CPK'] = rare_pred['predicted_cpk']
        result['RareLane_Price'] = rare_pred.get('predicted_cost')
        result['RareLane_Range'] = f"{rare_pred.get('cost_low', 0):,.0f} - {rare_pred.get('cost_high', 0):,.0f}"
        result['RareLane_Method'] = rare_pred['method']
        result['RareLane_Confidence'] = rare_pred['confidence']
    
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
commodities = sorted(df_knn['commodity'].unique())

# ============================================
# APP UI
# ============================================
st.title("🚚 Freight Pricing Negotiation Tool")
rare_lane_status = "✅ Rare Lane Model" if rare_lane_predictor else ""
st.caption(f"ML-powered pricing | Domestic | Default: Flatbed Trailer {rare_lane_status}")

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
        commodity_input = None if commodity_select == 'Auto-detect' else commodity_select

    with col3:
        weight = st.number_input("Weight (Tons)", min_value=0.0, max_value=100.0, value=0.0, step=1.0,
                                 help="Leave as 0 for auto-detect", key='single_weight')
        weight = None if weight == 0 else weight

    st.markdown("---")
    if st.button("🎯 Generate Pricing Corridor", type="primary", use_container_width=True, key='single_generate'):
        result = price_single_route(pickup_city, destination_city, vehicle_type, commodity_input, weight)
        lane_en = f"{pickup_en} → {dest_en}"
        lane_ar = f"{pickup_city} → {destination_city}"
        
        st.markdown("---")
        st.header("🎯 Pricing Corridor")
        st.info(f"**{lane_en}** | 🚛 {result['Vehicle_Type']} | 📏 {result['Distance_km']:.0f} km | ⚖️ {result['Weight_Tons']:.1f} T")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🟢 ANCHOR", f"{result['Anchor']:,.0f} SAR")
        with col2:
            st.metric("🟡 TARGET", f"{result['Target']:,.0f} SAR")
        with col3:
            st.metric("🔴 CEILING", f"{result['Ceiling']:,.0f} SAR")
        with col4:
            st.metric("💰 MARGIN", f"{result['Margin']:,.0f} SAR", f"{result['Margin_Pct']:.1f}%")
        
        st.caption(f"📊 Cost/km: **{result['Cost_Per_KM']:.2f} SAR** | Shipper: {result['Shipper_Rate']:,.0f} SAR | Source: {result['Recommendation_Source']}")
        
        # Rare Lane Model (if available)
        if rare_lane_predictor and 'RareLane_Price' in result:
            with st.expander("🔮 Rare Lane Model (for sparse lanes)", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Price", f"{result['RareLane_Price']:,.0f} SAR")
                with col2:
                    st.metric("Price Range", result['RareLane_Range'])
                with col3:
                    st.metric("Confidence", result['RareLane_Confidence'])
                st.caption(f"Method: {result['RareLane_Method']} | CPK: {result['RareLane_CPK']:.3f} SAR/km")
        
        # Historical & Recent Stats
        st.markdown("---")
        st.subheader(f"📊 Price History: {lane_en} ({result['Vehicle_Type']})")
        
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
        
        # AMMUNITION
        st.markdown("---")
        st.subheader("🚚 Your Ammunition (Recent Matches)")
        
        same_samples, other_samples = get_ammunition_loads(lane_ar, vehicle_type, commodity_input)
        
        commodity_used = commodity_input if commodity_input else result['Commodity']
        if len(same_samples) > 0:
            st.markdown(f"**Same Commodity ({commodity_used}):**")
            same_samples['Lane_EN'] = same_samples['pickup_city'].apply(to_english_city) + ' → ' + same_samples['destination_city'].apply(to_english_city)
            display_same = same_samples[['pickup_date', 'Lane_EN', 'commodity', 'total_carrier_price', 'days_ago']].copy()
            display_same.columns = ['Date', 'Lane', 'Commodity', 'Carrier (SAR)', 'Days Ago']
            display_same['Date'] = pd.to_datetime(display_same['Date']).dt.strftime('%Y-%m-%d')
            display_same['Carrier (SAR)'] = display_same['Carrier (SAR)'].round(0).astype(int)
            st.dataframe(display_same, use_container_width=True, hide_index=True)
        else:
            st.caption(f"No recent loads with {commodity_used}")
        
        if len(other_samples) > 0:
            st.markdown("**Other Commodities:**")
            other_samples['Lane_EN'] = other_samples['pickup_city'].apply(to_english_city) + ' → ' + other_samples['destination_city'].apply(to_english_city)
            display_other = other_samples[['pickup_date', 'Lane_EN', 'commodity', 'total_carrier_price', 'days_ago']].copy()
            display_other.columns = ['Date', 'Lane', 'Commodity', 'Carrier (SAR)', 'Days Ago']
            display_other['Date'] = pd.to_datetime(display_other['Date']).dt.strftime('%Y-%m-%d')
            display_other['Carrier (SAR)'] = display_other['Carrier (SAR)'].round(0).astype(int)
            st.dataframe(display_other, use_container_width=True, hide_index=True)
        else:
            st.caption("No recent loads with other commodities")
        
        total_shown = len(same_samples) + len(other_samples)
        if total_shown > 0:
            st.caption(f"**{total_shown} loads** | Samples ≥{MIN_DAYS_APART} days apart | Max {MAX_AGE_DAYS} days old")

# ============================================
# TAB 2: BULK ROUTE LOOKUP
# ============================================
with tab2:
    st.subheader("📦 Bulk Route Lookup")
    
    model_note = "+ Rare Lane Model predictions" if rare_lane_predictor else ""
    st.markdown(f"""
    **Upload a CSV to get historical and recent price stats for each route.**
    
    - **Required:** `From`, `To`
    - **Optional:** `Vehicle_Type` (default: Flatbed Trailer)
    - Output: Historical & Recent ({RECENCY_WINDOW}d) Min/Median/Max {model_note}
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
                    if not dest_matched and dest_ar not in VALID_CITIES_AR:
                        unmatched_cities.append({'Row': idx+1, 'Column': 'To', 'Original': dest_raw})
                    
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
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Routes", len(results_df))
                with col2:
                    with_data = (results_df['Hist_Count'] > 0).sum()
                    st.metric("With History", with_data)
                with col3:
                    with_recent = (results_df[f'Recent_{RECENCY_WINDOW}d_Count'] > 0).sum()
                    st.metric("With Recent", with_recent)
                
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

st.markdown("---")
st.caption("Freight Pricing Tool | Default: Flatbed Trailer | All prices in SAR")
