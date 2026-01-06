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
# FIXED SETTINGS (No sidebar)
# ============================================
ANCHOR_DISCOUNT = 12
CEILING_PREMIUM = 5
N_SIMILAR = 10
MIN_DAYS_APART = 5  # Samples should be at least 5 days apart

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
}

CITY_AR = {v: k for k, v in CITY_EN.items()}

CITY_MAPPING_EN_TO_AR = {
    'Jeddah': 'جدة', 'Jiddah': 'جدة', 'Jedda': 'جدة',
    'Riyadh': 'الرياض', 'Riyad': 'الرياض',
    'Dammam': 'الدمام', 'Dammam ': 'الدمام', 'Damam': 'الدمام',
    'Makkah': 'مكة المكرمة', 'Mecca': 'مكة المكرمة', 'Mekkah': 'مكة المكرمة', 'Mekkah ': 'مكة المكرمة',
    'Madina': 'المدينة المنورة', 'Madinah': 'المدينة المنورة', 'Medina': 'المدينة المنورة',
    'Rabigh': 'رابغ', 'Yanbu': 'ينبع', 'Yenbu': 'ينبع',
    'Tabuk': 'تبوك', 'Tabuk ': 'تبوك', 'Tabouk': 'تبوك',
    'Taif': 'الطائف', 'Tayef': 'الطائف',
    'Al Hasa': 'الْأَحْسَاء', 'Al-Hasa': 'الْأَحْسَاء', 'Ahsa': 'الْأَحْسَاء', 'Al Ahsa': 'الْأَحْسَاء',
    'Al Kharj': 'الخرج', 'Al-Kharij': 'الخرج', 'Kharj': 'الخرج',
    'Al Qassim': 'القصيم', 'Al-Qassim': 'القصيم', 'Al Qaseem': 'القصيم', 'Qassim': 'القصيم', 'Qaseem': 'القصيم',
    'Al Baha': 'ٱلْبَاحَة', 'Al-Baha': 'ٱلْبَاحَة', 'Baha': 'ٱلْبَاحَة',
    'Jizan': 'جازان', 'Jazan': 'جازان', 'Gizan': 'جازان', 'Sabya': 'جازان',
    'Najran': 'نجران', 'Nejran': 'نجران',
    'Abha': 'Abha', 'Abaha': 'Abha',
    'Khamis Mushait': 'Khamis Mushait', 'Khamis': 'Khamis Mushait', 'Khamis Mushit': 'Khamis Mushait',
    'Arar': 'عرعر', 'Arar ': 'عرعر',
    'Skaka': 'سكاكا', 'Sakaka': 'سكاكا',
    'Hafar Al Batin': 'حفر الباطن', 'Hafr Al Batin': 'حفر الباطن',
    'Neom': 'تبوك', 'NEOM': 'تبوك',
    'Sudair': 'سدير',
    'Umlug': 'Umluj', 'Umluj': 'Umluj',
    'Jeddah - 1': 'جدة', 'Jeddah - 2': 'جدة', 'Jeddah City': 'جدة',
    'Jeddah-1': 'جدة', 'Jeddah-2': 'جدة',
}

CITY_MAPPING_AR_TO_AR = {
    'جدة': 'جدة', 'جده': 'جدة',
    'الرياض': 'الرياض', 'رياض': 'الرياض',
    'الدمام': 'الدمام', 'دمام': 'الدمام',
    'مكة': 'مكة المكرمة', 'مكه': 'مكة المكرمة', 'مكة المكرمة': 'مكة المكرمة', 'مكه المكرمه': 'مكة المكرمة',
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
    'الاحساء': 'الْأَحْسَاء', 'الأحساء': 'الْأَحْسَاء', 'الْأَحْسَاء': 'الْأَحْسَاء', 'احساء': 'الْأَحْسَاء',
    'الباحة': 'ٱلْبَاحَة', 'الباحه': 'ٱلْبَاحَة', 'ٱلْبَاحَة': 'ٱلْبَاحَة', 'باحة': 'ٱلْبَاحَة',
    'حفر الباطن': 'حفر الباطن',
    'سدير': 'سدير',
    'ابها': 'Abha', 'أبها': 'Abha',
    'خميس مشيط': 'Khamis Mushait', 'خميس': 'Khamis Mushait',
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
    return {'carrier_model': carrier_model, 'shipper_model': shipper_model, 'config': config, 'df_knn': df_knn}

models = load_models()
config = models['config']
df_knn = models['df_knn']

FEATURES = config['FEATURES']
ENTITY_MAPPING = config.get('ENTITY_MAPPING', 'Domestic')
RECENCY_CUTOFF_DAYS = config.get('RECENCY_CUTOFF_DAYS', 90)
DISTANCE_LOOKUP = config.get('DISTANCE_LOOKUP', {})
BACKHAUL_LOOKUP = config.get('BACKHAUL_LOOKUP', {})
LANE_STATS = config.get('LANE_STATS', {})
RARE_LANE_THRESHOLD = config.get('RARE_LANE_THRESHOLD', 10)

df_knn = df_knn[df_knn['entity_mapping'] == ENTITY_MAPPING].copy()
VALID_CITIES_AR = set(df_knn['pickup_city'].unique()) | set(df_knn['destination_city'].unique())

# ============================================
# GET SIMILAR LOADS - EXACT LANE+VEHICLE MATCH
# ============================================
def get_similar_loads(lane, vehicle_ar, commodity=None, n_results=N_SIMILAR):
    """
    Get similar loads for ammunition.
    - Exact match on lane + vehicle type
    - Samples at least MIN_DAYS_APART days apart
    - Diversify commodities if commodity not specified
    """
    # Filter to exact lane + vehicle match
    matches = df_knn[
        (df_knn['lane'] == lane) & 
        (df_knn['vehicle_type'] == vehicle_ar)
    ].copy()
    
    if len(matches) == 0:
        return pd.DataFrame()
    
    # Sort by recency
    matches = matches.sort_values('days_ago', ascending=True)
    
    # Select samples at least MIN_DAYS_APART apart
    selected = []
    last_days_ago = -MIN_DAYS_APART  # Start so first one is always selected
    
    if commodity and commodity not in ['', 'Auto', 'auto', None]:
        # Commodity specified - just pick samples MIN_DAYS_APART
        for _, row in matches.iterrows():
            if row['days_ago'] >= last_days_ago + MIN_DAYS_APART:
                selected.append(row)
                last_days_ago = row['days_ago']
                if len(selected) >= n_results:
                    break
    else:
        # Commodity NOT specified - diversify commodities
        commodities_seen = set()
        
        # First pass: get one sample per commodity (at least MIN_DAYS_APART)
        for _, row in matches.iterrows():
            if row['days_ago'] >= last_days_ago + MIN_DAYS_APART:
                if row['commodity'] not in commodities_seen:
                    selected.append(row)
                    commodities_seen.add(row['commodity'])
                    last_days_ago = row['days_ago']
                    if len(selected) >= n_results:
                        break
        
        # Second pass: fill remaining slots with any commodity (still MIN_DAYS_APART)
        if len(selected) < n_results:
            last_days_ago = selected[-1]['days_ago'] if selected else -MIN_DAYS_APART
            for _, row in matches.iterrows():
                if row['days_ago'] >= last_days_ago + MIN_DAYS_APART:
                    # Check if this exact row is already selected
                    if not any((s['pickup_date'] == row['pickup_date'] and 
                               s['total_carrier_price'] == row['total_carrier_price']) for s in selected):
                        selected.append(row)
                        last_days_ago = row['days_ago']
                        if len(selected) >= n_results:
                            break
    
    if len(selected) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(selected)

# ============================================
# BULK PRICING FUNCTION
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
    
    recent_data = lane_data[lane_data['days_ago'] <= RECENCY_CUTOFF_DAYS]
    
    is_same_city = (pickup_ar == dest_ar)
    if is_same_city and len(lane_data) > 0:
        local_data = lane_data[lane_data['is_multistop'] == 0]
        local_recent = local_data[local_data['days_ago'] <= RECENCY_CUTOFF_DAYS]
        if len(local_data) > 0:
            lane_data = local_data
            recent_data = local_recent
    
    if len(lane_data) > 0:
        hist_count = len(lane_data)
        hist_min = lane_data['total_carrier_price'].min()
        hist_max = lane_data['total_carrier_price'].max()
        hist_median = lane_data['total_carrier_price'].median()
    else:
        hist_count = 0
        hist_min = hist_max = hist_median = None
    
    if len(recent_data) > 0:
        recent_count = len(recent_data)
        recent_min = recent_data['total_carrier_price'].min()
        recent_max = recent_data['total_carrier_price'].max()
        recent_median = recent_data['total_carrier_price'].median()
    else:
        recent_count = 0
        recent_min = recent_max = recent_median = None
    
    if commodity is None or commodity in ['', 'Auto', 'auto']:
        if len(lane_data) > 0:
            commodity = lane_data['commodity'].mode().iloc[0]
        else:
            commodity = df_knn['commodity'].mode().iloc[0]
    
    if weight is None or weight == 0:
        comm_weights = df_knn[(df_knn['commodity'] == commodity) & (df_knn['weight'] > 0)]['weight']
        weight = comm_weights.median() if len(comm_weights) > 0 else df_knn['weight'].median()
    
    if len(lane_data) > 0:
        container = int(lane_data['container'].mode().iloc[0])
    else:
        container = 0
    
    if len(lane_data) > 0:
        distance = lane_data['distance'].median()
    elif lane in DISTANCE_LOOKUP:
        distance = DISTANCE_LOOKUP[lane]
    else:
        similar = [l for l in DISTANCE_LOOKUP.keys() if l.startswith(pickup_ar + ' →')]
        distance = np.mean([DISTANCE_LOOKUP[l] for l in similar]) if similar else 500
    
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
    
    return {
        'Vehicle_Type': to_english_vehicle(vehicle_ar),
        'Commodity': commodity,
        'Weight_Tons': round(weight, 1),
        'Distance_km': round(distance, 0),
        'Hist_Count': hist_count,
        'Hist_Min': round(hist_min, 0) if hist_min else None,
        'Hist_Median': round(hist_median, 0) if hist_median else None,
        'Hist_Max': round(hist_max, 0) if hist_max else None,
        f'Recent_{RECENCY_CUTOFF_DAYS}d_Count': recent_count,
        f'Recent_{RECENCY_CUTOFF_DAYS}d_Min': round(recent_min, 0) if recent_min else None,
        f'Recent_{RECENCY_CUTOFF_DAYS}d_Median': round(recent_median, 0) if recent_median else None,
        f'Recent_{RECENCY_CUTOFF_DAYS}d_Max': round(recent_max, 0) if recent_max else None,
        'Model_Prediction': round(pred_carrier, 0),
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
st.caption(f"ML-powered pricing | Domestic | Default: Flatbed Trailer")

tab1, tab2 = st.tabs(["🎯 Single Route Pricing", "📦 Bulk CSV Pricing"])

# ============================================
# TAB 1: SINGLE ROUTE PRICING
# ============================================
with tab1:
    st.subheader("📋 Route Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        pickup_en = st.selectbox(
            "Pickup City",
            options=pickup_cities_en,
            index=pickup_cities_en.index('Jeddah') if 'Jeddah' in pickup_cities_en else 0,
            key='single_pickup'
        )
        pickup_city = to_arabic_city(pickup_en)

    with col2:
        dest_en = st.selectbox(
            "Destination City",
            options=dest_cities_en,
            index=dest_cities_en.index('Riyadh') if 'Riyadh' in dest_cities_en else 0,
            key='single_dest'
        )
        destination_city = to_arabic_city(dest_en)

    with col3:
        default_idx = vehicle_types_en.index(DEFAULT_VEHICLE_EN) if DEFAULT_VEHICLE_EN in vehicle_types_en else 0
        vehicle_en = st.selectbox(
            "Vehicle Type",
            options=vehicle_types_en,
            index=default_idx,
            key='single_vehicle'
        )
        vehicle_type = to_arabic_vehicle(vehicle_en)

    st.subheader("📦 Optional Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        container_options = ['Auto-detect', 'Yes (Container)', 'No (Non-container)']
        container_select = st.selectbox("Container", options=container_options, key='single_container')
        container = None if container_select == 'Auto-detect' else (1 if 'Yes' in container_select else 0)

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
            st.metric("🟢 ANCHOR", f"{result['Anchor']:,.0f} SAR", help="Start negotiation here")
        with col2:
            st.metric("🟡 TARGET", f"{result['Target']:,.0f} SAR", help="Fair market price")
        with col3:
            st.metric("🔴 CEILING", f"{result['Ceiling']:,.0f} SAR", help="Maximum acceptable")
        with col4:
            st.metric("💰 MARGIN", f"{result['Margin']:,.0f} SAR", f"{result['Margin_Pct']:.1f}%")
        
        st.caption(f"📊 Cost/km: **{result['Cost_Per_KM']:.2f} SAR** | Shipper: {result['Shipper_Rate']:,.0f} SAR | Source: {result['Recommendation_Source']}")
        
        # Historical & Recent Stats
        st.markdown("---")
        st.subheader(f"📊 Price History: {lane_en} ({result['Vehicle_Type']})")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Historical (All Time)**")
            if result['Hist_Count'] > 0:
                hist_df = pd.DataFrame({
                    'Metric': ['Count', 'Min', 'Median', 'Max'],
                    'Value': [f"{result['Hist_Count']} loads", f"{result['Hist_Min']:,.0f} SAR",
                              f"{result['Hist_Median']:,.0f} SAR", f"{result['Hist_Max']:,.0f} SAR"]
                })
                st.dataframe(hist_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No historical data")
        
        with col2:
            st.markdown(f"**Recent ({RECENCY_CUTOFF_DAYS} Days)**")
            if result[f'Recent_{RECENCY_CUTOFF_DAYS}d_Count'] > 0:
                recent_df = pd.DataFrame({
                    'Metric': ['Count', 'Min', 'Median', 'Max'],
                    'Value': [f"{result[f'Recent_{RECENCY_CUTOFF_DAYS}d_Count']} loads",
                              f"{result[f'Recent_{RECENCY_CUTOFF_DAYS}d_Min']:,.0f} SAR",
                              f"{result[f'Recent_{RECENCY_CUTOFF_DAYS}d_Median']:,.0f} SAR",
                              f"{result[f'Recent_{RECENCY_CUTOFF_DAYS}d_Max']:,.0f} SAR"]
                })
                st.dataframe(recent_df, use_container_width=True, hide_index=True)
            else:
                st.warning(f"No loads in last {RECENCY_CUTOFF_DAYS} days")
        
        # Similar Loads - EXACT MATCH
        st.markdown("---")
        st.subheader("🚚 Your Ammunition (Exact Lane + Vehicle Match)")
        
        similar = get_similar_loads(lane_ar, vehicle_type, commodity_input)
        
        if len(similar) > 0:
            similar['margin'] = similar['total_shipper_price'] - similar['total_carrier_price']
            similar['Lane_EN'] = similar['pickup_city'].apply(to_english_city) + ' → ' + similar['destination_city'].apply(to_english_city)
            similar['Vehicle_EN'] = similar['vehicle_type'].apply(to_english_vehicle)
            
            display_df = similar[['pickup_date', 'Lane_EN', 'Vehicle_EN', 'commodity', 'distance', 'total_carrier_price', 'margin', 'days_ago']].copy()
            display_df.columns = ['Date', 'Lane', 'Vehicle', 'Commodity', 'Distance (km)', 'Carrier (SAR)', 'Margin (SAR)', 'Days Ago']
            display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
            display_df['Distance (km)'] = display_df['Distance (km)'].round(0).astype(int)
            display_df['Carrier (SAR)'] = display_df['Carrier (SAR)'].round(0).astype(int)
            display_df['Margin (SAR)'] = display_df['Margin (SAR)'].round(0).astype(int)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            unique_commodities = similar['commodity'].nunique()
            st.caption(f"**{len(similar)} loads** | {unique_commodities} different commodities | Samples ≥{MIN_DAYS_APART} days apart")
        else:
            st.warning(f"No exact matches found for {lane_en} with {result['Vehicle_Type']}")

# ============================================
# TAB 2: BULK CSV PRICING
# ============================================
with tab2:
    st.subheader("📦 Bulk Pricing from CSV")
    
    st.markdown(f"""
    **Upload a CSV with your routes.**
    
    **Required:** `From`, `To` (English or Arabic)
    
    **Optional:** `Vehicle_Type`, `Commodity`, `Weight`, `Monthly_Trips`
    
    - Empty = auto-detect | Default vehicle: **Flatbed Trailer**
    """)
    
    sample_df = pd.DataFrame({
        'From': ['Jeddah', 'Jeddah', 'Riyadh', 'جدة'],
        'To': ['Riyadh', 'Dammam', 'Jeddah', 'الرياض'],
        'Vehicle_Type': ['Flatbed Trailer', '', '', ''],
        'Commodity': ['', '', '', ''],
        'Weight': [28, 25, 0, 28],
        'Monthly_Trips': [100, 50, 80, 150],
    })
    
    st.download_button("📥 Download Template", sample_df.to_csv(index=False), "pricing_template.csv", "text/csv")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            routes_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(routes_df)} routes")
            
            with st.expander("📋 Preview"):
                st.dataframe(routes_df.head(10), use_container_width=True)
            
            if st.button("🚀 Price All Routes", type="primary", use_container_width=True):
                results = []
                unmatched_cities = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in routes_df.iterrows():
                    pickup_raw = str(row.get('From', '')).strip()
                    dest_raw = str(row.get('To', '')).strip()
                    vehicle_raw = str(row.get('Vehicle_Type', '')).strip() if 'Vehicle_Type' in row else ''
                    commodity = str(row.get('Commodity', '')).strip() if 'Commodity' in row else None
                    weight = row.get('Weight', 0) if 'Weight' in row else None
                    trips = row.get('Monthly_Trips', 1) if 'Monthly_Trips' in row else 1
                    
                    pickup_ar, pickup_matched = normalize_city(pickup_raw)
                    dest_ar, dest_matched = normalize_city(dest_raw)
                    
                    if not pickup_matched and pickup_ar not in VALID_CITIES_AR:
                        unmatched_cities.append({'Row': idx+1, 'Column': 'From', 'Original': pickup_raw, 'Normalized': pickup_ar})
                    if not dest_matched and dest_ar not in VALID_CITIES_AR:
                        unmatched_cities.append({'Row': idx+1, 'Column': 'To', 'Original': dest_raw, 'Normalized': dest_ar})
                    
                    if vehicle_raw in ['', 'nan', 'None', 'Auto', 'auto']:
                        vehicle_ar = DEFAULT_VEHICLE_AR
                    else:
                        vehicle_ar = to_arabic_vehicle(vehicle_raw)
                    
                    if commodity in ['', 'nan', 'None', 'Auto', 'auto']:
                        commodity = None
                    if pd.isna(weight) or weight == 0:
                        weight = None
                    if pd.isna(trips):
                        trips = 1
                    
                    status_text.text(f"Processing {idx+1}/{len(routes_df)}: {pickup_raw} → {dest_raw}")
                    
                    result = price_single_route(pickup_ar, dest_ar, vehicle_ar, commodity, weight)
                    
                    result['From_Original'] = pickup_raw
                    result['To_Original'] = dest_raw
                    result['From'] = to_english_city(pickup_ar) if pickup_ar else pickup_raw
                    result['To'] = to_english_city(dest_ar) if dest_ar else dest_raw
                    result['Monthly_Trips'] = trips
                    result['Monthly_Carrier_Cost'] = result['Recommended_Carrier'] * trips
                    result['Monthly_Revenue'] = result['Shipper_Rate'] * trips
                    result['Monthly_Margin'] = result['Monthly_Revenue'] - result['Monthly_Carrier_Cost']
                    
                    results.append(result)
                    progress_bar.progress((idx + 1) / len(routes_df))
                
                status_text.text("✅ Complete!")
                results_df = pd.DataFrame(results)
                
                col_order = [
                    'From', 'To', 'Vehicle_Type', 'Commodity', 'Weight_Tons', 'Distance_km', 'Monthly_Trips',
                    'Hist_Count', 'Hist_Min', 'Hist_Median', 'Hist_Max',
                    f'Recent_{RECENCY_CUTOFF_DAYS}d_Count', f'Recent_{RECENCY_CUTOFF_DAYS}d_Min', 
                    f'Recent_{RECENCY_CUTOFF_DAYS}d_Median', f'Recent_{RECENCY_CUTOFF_DAYS}d_Max',
                    'Model_Prediction', 'Recommended_Carrier', 'Recommendation_Source',
                    'Anchor', 'Target', 'Ceiling',
                    'Shipper_Rate', 'Cost_Per_KM', 'Margin', 'Margin_Pct',
                    'Monthly_Carrier_Cost', 'Monthly_Revenue', 'Monthly_Margin'
                ]
                results_df = results_df[[c for c in col_order if c in results_df.columns]]
                
                st.markdown("---")
                st.subheader("📊 Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Routes", len(results_df))
                with col2:
                    st.metric("With History", (results_df['Hist_Count'] > 0).sum())
                with col3:
                    st.metric("With Recent", (results_df[f'Recent_{RECENCY_CUTOFF_DAYS}d_Count'] > 0).sum())
                with col4:
                    st.metric("Monthly Margin", f"{results_df['Monthly_Margin'].sum():,.0f} SAR")
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                st.download_button("📥 Download Results", results_df.to_csv(index=False), 
                                  "pricing_results.csv", "text/csv", type="primary")
                
                if len(unmatched_cities) > 0:
                    st.markdown("---")
                    st.subheader("⚠️ Unmatched Cities")
                    st.warning(f"{len(unmatched_cities)} cities could not be matched. Check spelling.")
                    st.dataframe(pd.DataFrame(unmatched_cities), use_container_width=True, hide_index=True)
                
                with st.expander("📈 Summary"):
                    st.markdown("**By Source:**")
                    st.dataframe(results_df.groupby('Recommendation_Source').agg({
                        'From': 'count', 'Monthly_Margin': 'sum'
                    }).rename(columns={'From': 'Routes'}), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("Freight Pricing Tool | Default: Flatbed Trailer | All prices in SAR")
