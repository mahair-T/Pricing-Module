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

# Default vehicle type
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

# Extended city mapping for CSV uploads (English -> Arabic)
CITY_MAPPING_EN_TO_AR = {
    # English variants
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

# Arabic variants mapping (Arabic -> Standard Arabic)
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
    """
    Normalize city name to standard Arabic format.
    Handles English, Arabic, and various spellings.
    Returns (normalized_city, was_matched)
    """
    if pd.isna(city_raw) or city_raw == '':
        return None, False
    
    city = str(city_raw).strip()
    
    # Remove extra whitespace
    city = re.sub(r'\s+', ' ', city).strip()
    
    # Check if it's Arabic (contains Arabic characters)
    is_arabic = bool(re.search(r'[\u0600-\u06FF]', city))
    
    if is_arabic:
        # Try Arabic mapping
        if city in CITY_MAPPING_AR_TO_AR:
            return CITY_MAPPING_AR_TO_AR[city], True
        # Try without diacritics (simplified check)
        for ar_variant, standard in CITY_MAPPING_AR_TO_AR.items():
            if ar_variant in city or city in ar_variant:
                return standard, True
        # Check if it's already a valid city in our data
        return city, False  # Return as-is, mark as unmatched for validation
    else:
        # English - try mapping
        # Case-insensitive lookup
        city_lower = city.lower()
        for en_variant, ar_standard in CITY_MAPPING_EN_TO_AR.items():
            if en_variant.lower() == city_lower:
                return ar_standard, True
        # Try partial match
        for en_variant, ar_standard in CITY_MAPPING_EN_TO_AR.items():
            if en_variant.lower() in city_lower or city_lower in en_variant.lower():
                return ar_standard, True
        return city, False  # Return as-is, mark as unmatched

def to_english_city(city_ar):
    return CITY_EN.get(city_ar, city_ar)

def to_arabic_city(city_en):
    # Try direct mapping first, then extended
    if city_en in CITY_AR:
        return CITY_AR[city_en]
    normalized, _ = normalize_city(city_en)
    return normalized if normalized else city_en

def to_english_vehicle(vtype_ar):
    return VEHICLE_TYPE_EN.get(vtype_ar, vtype_ar)

def to_arabic_vehicle(vtype_en):
    if vtype_en in VEHICLE_TYPE_AR:
        return VEHICLE_TYPE_AR[vtype_en]
    # Handle Arabic input
    if vtype_en in VEHICLE_TYPE_EN:
        return vtype_en
    return DEFAULT_VEHICLE_AR  # Default to flatbed

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
    
    return {
        'carrier_model': carrier_model,
        'shipper_model': shipper_model,
        'config': config,
        'df_knn': df_knn
    }

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

# Get valid cities from data for validation
VALID_CITIES_AR = set(df_knn['pickup_city'].unique()) | set(df_knn['destination_city'].unique())

# ============================================
# BULK PRICING FUNCTION
# ============================================
def price_single_route(pickup_ar, dest_ar, vehicle_ar=None, commodity=None, weight=None, 
                       anchor_discount=12, ceiling_premium=5):
    """Price a single route and return all stats."""
    
    # Default to flatbed if not specified
    if vehicle_ar is None or vehicle_ar in ['', 'Auto', 'auto', None]:
        vehicle_ar = DEFAULT_VEHICLE_AR
    
    lane = f"{pickup_ar} → {dest_ar}"
    
    # Get base data - filter by vehicle type
    lane_data = df_knn[(df_knn['lane'] == lane) & (df_knn['vehicle_type'] == vehicle_ar)].copy()
    
    # Further filter by commodity if specified
    if commodity and commodity not in ['', 'Auto', 'auto', None]:
        lane_commodity_data = lane_data[lane_data['commodity'] == commodity]
        if len(lane_commodity_data) > 0:
            lane_data = lane_commodity_data
    
    recent_data = lane_data[lane_data['days_ago'] <= RECENCY_CUTOFF_DAYS]
    
    # Handle local delivery
    is_same_city = (pickup_ar == dest_ar)
    if is_same_city and len(lane_data) > 0:
        local_data = lane_data[lane_data['is_multistop'] == 0]
        local_recent = local_data[local_data['days_ago'] <= RECENCY_CUTOFF_DAYS]
        if len(local_data) > 0:
            lane_data = local_data
            recent_data = local_recent
    
    # Historical stats
    if len(lane_data) > 0:
        hist_count = len(lane_data)
        hist_min = lane_data['total_carrier_price'].min()
        hist_max = lane_data['total_carrier_price'].max()
        hist_median = lane_data['total_carrier_price'].median()
    else:
        hist_count = 0
        hist_min = hist_max = hist_median = None
    
    # Recent stats
    if len(recent_data) > 0:
        recent_count = len(recent_data)
        recent_min = recent_data['total_carrier_price'].min()
        recent_max = recent_data['total_carrier_price'].max()
        recent_median = recent_data['total_carrier_price'].median()
    else:
        recent_count = 0
        recent_min = recent_max = recent_median = None
    
    # Auto-detect commodity if not specified
    if commodity is None or commodity in ['', 'Auto', 'auto']:
        if len(lane_data) > 0:
            commodity = lane_data['commodity'].mode().iloc[0]
        else:
            commodity = df_knn['commodity'].mode().iloc[0]
    
    # Auto-detect weight if not specified
    if weight is None or weight == 0:
        comm_weights = df_knn[(df_knn['commodity'] == commodity) & (df_knn['weight'] > 0)]['weight']
        weight = comm_weights.median() if len(comm_weights) > 0 else df_knn['weight'].median()
    
    # Container
    if len(lane_data) > 0:
        container = int(lane_data['container'].mode().iloc[0])
    else:
        container = 0
    
    # Distance
    if len(lane_data) > 0:
        distance = lane_data['distance'].median()
    elif lane in DISTANCE_LOOKUP:
        distance = DISTANCE_LOOKUP[lane]
    else:
        similar = [l for l in DISTANCE_LOOKUP.keys() if l.startswith(pickup_ar + ' →')]
        distance = np.mean([DISTANCE_LOOKUP[l] for l in similar]) if similar else 500
    
    # Multistop
    is_multistop = 0 if is_same_city else (int(lane_data['is_multistop'].mode().iloc[0]) if len(lane_data) > 0 else 0)
    
    # Model prediction
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
    
    # Blend with actuals
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
    
    # Corridor
    anchor = recommended * (1 - anchor_discount / 100)
    target = recommended
    ceiling = recommended * (1 + ceiling_premium / 100)
    cost_per_km = recommended / distance if distance > 0 else None
    
    return {
        'Vehicle_Type': to_english_vehicle(vehicle_ar),
        'Commodity': commodity,
        'Weight_Tons': round(weight, 1),
        'Distance_km': round(distance, 0),
        
        # Historical
        'Hist_Count': hist_count,
        'Hist_Min': round(hist_min, 0) if hist_min else None,
        'Hist_Median': round(hist_median, 0) if hist_median else None,
        'Hist_Max': round(hist_max, 0) if hist_max else None,
        
        # Recent
        f'Recent_{RECENCY_CUTOFF_DAYS}d_Count': recent_count,
        f'Recent_{RECENCY_CUTOFF_DAYS}d_Min': round(recent_min, 0) if recent_min else None,
        f'Recent_{RECENCY_CUTOFF_DAYS}d_Median': round(recent_median, 0) if recent_median else None,
        f'Recent_{RECENCY_CUTOFF_DAYS}d_Max': round(recent_max, 0) if recent_max else None,
        
        # Model & Recommendation
        'Model_Prediction': round(pred_carrier, 0),
        'Recommended_Carrier': round(recommended, 0),
        'Recommendation_Source': source,
        'Shipper_Rate': round(pred_shipper, 0),
        'Cost_Per_KM': round(cost_per_km, 2) if cost_per_km else None,
        
        # Corridor
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
# APP UI - TABS
# ============================================
st.title("🚚 Freight Pricing Negotiation Tool")
st.caption(f"ML-powered pricing | Domestic | {RECENCY_CUTOFF_DAYS}-day recency window | Default: Flatbed Trailer")

tab1, tab2 = st.tabs(["🎯 Single Route Pricing", "📦 Bulk CSV Pricing"])

# ============================================
# TAB 1: SINGLE ROUTE PRICING
# ============================================
with tab1:
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        anchor_discount = st.slider("Anchor Discount %", 5, 20, 12, help="Discount from target for starting price")
        ceiling_premium = st.slider("Ceiling Premium %", 3, 15, 5, help="Premium above target for maximum price")
        n_similar = st.slider("Similar Loads to Show", 5, 20, 10)
        
        st.markdown("---")
        st.markdown("**📊 Data Summary**")
        st.caption(f"Total loads: {len(df_knn):,}")
        st.caption(f"Unique lanes: {df_knn['lane'].nunique()}")
        st.caption(f"Default vehicle: {DEFAULT_VEHICLE_EN}")

    # Main inputs
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
        # Default to Flatbed Trailer
        default_idx = vehicle_types_en.index(DEFAULT_VEHICLE_EN) if DEFAULT_VEHICLE_EN in vehicle_types_en else 0
        vehicle_en = st.selectbox(
            "Vehicle Type",
            options=vehicle_types_en,
            index=default_idx,
            key='single_vehicle'
        )
        vehicle_type = to_arabic_vehicle(vehicle_en)

    # Optional inputs
    st.subheader("📦 Optional Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        container_options = ['Auto-detect', 'Yes (Container)', 'No (Non-container)']
        container_select = st.selectbox("Container", options=container_options, key='single_container')
        container = None if container_select == 'Auto-detect' else (1 if 'Yes' in container_select else 0)

    with col2:
        commodity_options = ['Auto-detect'] + commodities
        commodity_select = st.selectbox("Commodity", options=commodity_options, key='single_commodity')
        commodity = None if commodity_select == 'Auto-detect' else commodity_select

    with col3:
        weight = st.number_input("Weight (Tons)", min_value=0.0, max_value=100.0, value=0.0, step=1.0,
                                 help="Leave as 0 for auto-detect", key='single_weight')
        weight = None if weight == 0 else weight

    # Generate button
    st.markdown("---")
    if st.button("🎯 Generate Pricing Corridor", type="primary", use_container_width=True, key='single_generate'):
        
        result = price_single_route(pickup_city, destination_city, vehicle_type, commodity, weight, anchor_discount, ceiling_premium)
        
        lane_en = f"{pickup_en} → {dest_en}"
        
        # Display results
        st.markdown("---")
        st.header("🎯 Pricing Corridor")
        
        # Route info bar
        st.info(f"**{lane_en}** | 🚛 {result['Vehicle_Type']} | 📏 {result['Distance_km']:.0f} km | ⚖️ {result['Weight_Tons']:.1f} T")
        
        # Corridor metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🟢 ANCHOR", f"{result['Anchor']:,.0f} SAR", help="Start negotiation here")
        with col2:
            st.metric("🟡 TARGET", f"{result['Target']:,.0f} SAR", help="Fair market price")
        with col3:
            st.metric("🔴 CEILING", f"{result['Ceiling']:,.0f} SAR", help="Maximum acceptable")
        with col4:
            st.metric("💰 MARGIN", f"{result['Margin']:,.0f} SAR", f"{result['Margin_Pct']:.1f}%")
        
        # Cost per km
        st.caption(f"📊 Cost per km: **{result['Cost_Per_KM']:.2f} SAR/km** | Shipper rate: {result['Shipper_Rate']:,.0f} SAR | Source: {result['Recommendation_Source']}")
        
        # Historical & Recent Stats
        st.markdown("---")
        st.subheader(f"📊 Price History: {lane_en} ({result['Vehicle_Type']})")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Historical (All Time)**")
            if result['Hist_Count'] > 0:
                hist_df = pd.DataFrame({
                    'Metric': ['Count', 'Min', 'Median', 'Max'],
                    'Value': [
                        f"{result['Hist_Count']} loads",
                        f"{result['Hist_Min']:,.0f} SAR",
                        f"{result['Hist_Median']:,.0f} SAR",
                        f"{result['Hist_Max']:,.0f} SAR",
                    ]
                })
                st.dataframe(hist_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No historical data for this lane + vehicle")
        
        with col2:
            st.markdown(f"**Recent ({RECENCY_CUTOFF_DAYS} Days)**")
            if result[f'Recent_{RECENCY_CUTOFF_DAYS}d_Count'] > 0:
                recent_df = pd.DataFrame({
                    'Metric': ['Count', 'Min', 'Median', 'Max'],
                    'Value': [
                        f"{result[f'Recent_{RECENCY_CUTOFF_DAYS}d_Count']} loads",
                        f"{result[f'Recent_{RECENCY_CUTOFF_DAYS}d_Min']:,.0f} SAR",
                        f"{result[f'Recent_{RECENCY_CUTOFF_DAYS}d_Median']:,.0f} SAR",
                        f"{result[f'Recent_{RECENCY_CUTOFF_DAYS}d_Max']:,.0f} SAR",
                    ]
                })
                st.dataframe(recent_df, use_container_width=True, hide_index=True)
            else:
                st.warning(f"No loads in last {RECENCY_CUTOFF_DAYS} days")
        
        # Similar Loads
        st.markdown("---")
        st.subheader("🚚 Similar Loads (Your Ammunition)")
        
        lane = f"{pickup_city} → {destination_city}"
        
        base_data = df_knn[
            (df_knn['vehicle_type'] == vehicle_type) &
            (df_knn['days_ago'] <= RECENCY_CUTOFF_DAYS)
        ].copy()
        
        if len(base_data) == 0:
            base_data = df_knn[df_knn['vehicle_type'] == vehicle_type].copy()
            st.warning(f"⏳ No loads in last {RECENCY_CUTOFF_DAYS} days - showing older data")
        
        base_data['is_same_lane'] = base_data['lane'] == lane
        base_data['is_same_commodity'] = base_data['commodity'] == result['Commodity']
        
        base_data['match_score'] = (
            base_data['is_same_lane'].astype(int) * 100 +
            base_data['is_same_commodity'].astype(int) * 10 -
            base_data['days_ago'] * 0.5
        )
        
        base_data = base_data.sort_values(['match_score', 'days_ago'], ascending=[False, True])
        similar = base_data.head(n_similar).copy()
        
        if len(similar) > 0:
            similar['margin'] = similar['total_shipper_price'] - similar['total_carrier_price']
            
            def get_match_label(row):
                parts = []
                if row['is_same_lane']: parts.append('🎯')
                if row['is_same_commodity']: parts.append('📦')
                if row['days_ago'] <= 30: parts.append('🔥')
                elif row['days_ago'] <= 60: parts.append('✨')
                return ' '.join(parts) if parts else '📋'
            
            similar['Match'] = similar.apply(get_match_label, axis=1)
            similar['Lane_EN'] = similar['pickup_city'].apply(to_english_city) + ' → ' + similar['destination_city'].apply(to_english_city)
            similar['Vehicle_EN'] = similar['vehicle_type'].apply(to_english_vehicle)
            
            display_df = similar[['Match', 'pickup_date', 'Lane_EN', 'Vehicle_EN', 'commodity', 'distance', 'total_carrier_price', 'margin', 'days_ago']].copy()
            display_df.columns = ['Match', 'Date', 'Lane', 'Vehicle', 'Commodity', 'Distance (km)', 'Carrier (SAR)', 'Margin (SAR)', 'Days Ago']
            display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
            display_df['Distance (km)'] = display_df['Distance (km)'].round(0).astype(int)
            display_df['Carrier (SAR)'] = display_df['Carrier (SAR)'].round(0).astype(int)
            display_df['Margin (SAR)'] = display_df['Margin (SAR)'].round(0).astype(int)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            same_lane_n = similar['is_same_lane'].sum()
            very_recent = (similar['days_ago'] <= 30).sum()
            st.caption(f"**{len(similar)} loads shown** | 🎯 Same lane: {same_lane_n} | 🔥 Last 30 days: {very_recent}")
            st.caption("**Legend:** 🎯 = Same lane | 📦 = Same commodity | 🔥 = Last 30 days | ✨ = Last 60 days")
        else:
            st.warning("No similar loads found")

# ============================================
# TAB 2: BULK CSV PRICING
# ============================================
with tab2:
    st.subheader("📦 Bulk Pricing from CSV")
    
    st.markdown("""
    **Upload a CSV file with your routes to price them all at once.**
    
    **Required columns:** `From`, `To` (English or Arabic city names supported)
    
    **Optional columns:** `Vehicle_Type`, `Commodity`, `Weight`, `Monthly_Trips`
    
    - Leave cells empty or use 'Auto' for auto-detection
    - **Default vehicle type: Flatbed Trailer** (if not specified)
    - Arabic and English city names are automatically normalized
    """)
    
    # Download sample template
    sample_df = pd.DataFrame({
        'From': ['Jeddah', 'Jeddah', 'Riyadh', 'جدة'],
        'To': ['Riyadh', 'Dammam', 'Jeddah', 'الرياض'],
        'Vehicle_Type': ['Flatbed Trailer', '', '', ''],
        'Commodity': ['', '', '', ''],
        'Weight': [28, 25, 0, 28],
        'Monthly_Trips': [100, 50, 80, 150],
    })
    
    csv_template = sample_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample Template",
        data=csv_template,
        file_name="pricing_template.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    # Settings for bulk
    col1, col2 = st.columns(2)
    with col1:
        bulk_anchor_discount = st.number_input("Anchor Discount %", min_value=0, max_value=30, value=12, key='bulk_anchor')
    with col2:
        bulk_ceiling_premium = st.number_input("Ceiling Premium %", min_value=0, max_value=30, value=5, key='bulk_ceiling')
    
    if uploaded_file is not None:
        try:
            routes_df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(routes_df)} routes")
            
            # Show preview
            with st.expander("📋 Preview uploaded data"):
                st.dataframe(routes_df.head(10), use_container_width=True)
            
            # Process button
            if st.button("🚀 Price All Routes", type="primary", use_container_width=True):
                
                results = []
                unmatched_cities = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in routes_df.iterrows():
                    # Extract values
                    pickup_raw = str(row.get('From', '')).strip()
                    dest_raw = str(row.get('To', '')).strip()
                    vehicle_raw = str(row.get('Vehicle_Type', '')).strip() if 'Vehicle_Type' in row else ''
                    commodity = str(row.get('Commodity', '')).strip() if 'Commodity' in row else None
                    weight = row.get('Weight', 0) if 'Weight' in row else None
                    trips = row.get('Monthly_Trips', 1) if 'Monthly_Trips' in row else 1
                    
                    # Normalize city names
                    pickup_ar, pickup_matched = normalize_city(pickup_raw)
                    dest_ar, dest_matched = normalize_city(dest_raw)
                    
                    # Track unmatched
                    if not pickup_matched and pickup_ar not in VALID_CITIES_AR:
                        unmatched_cities.append({'Row': idx+1, 'Column': 'From', 'Original': pickup_raw, 'Normalized': pickup_ar})
                    if not dest_matched and dest_ar not in VALID_CITIES_AR:
                        unmatched_cities.append({'Row': idx+1, 'Column': 'To', 'Original': dest_raw, 'Normalized': dest_ar})
                    
                    # Handle vehicle type - default to flatbed
                    if vehicle_raw in ['', 'nan', 'None', 'Auto', 'auto']:
                        vehicle_ar = DEFAULT_VEHICLE_AR
                    else:
                        vehicle_ar = to_arabic_vehicle(vehicle_raw)
                    
                    # Handle other fields
                    if commodity in ['', 'nan', 'None', 'Auto', 'auto']:
                        commodity = None
                    if pd.isna(weight) or weight == 0:
                        weight = None
                    if pd.isna(trips):
                        trips = 1
                    
                    status_text.text(f"Processing {idx+1}/{len(routes_df)}: {pickup_raw} → {dest_raw}")
                    
                    # Price the route
                    result = price_single_route(pickup_ar, dest_ar, vehicle_ar, commodity, weight, 
                                               bulk_anchor_discount, bulk_ceiling_premium)
                    
                    # Add original and normalized names
                    result['From_Original'] = pickup_raw
                    result['To_Original'] = dest_raw
                    result['From'] = to_english_city(pickup_ar) if pickup_ar else pickup_raw
                    result['To'] = to_english_city(dest_ar) if dest_ar else dest_raw
                    
                    # Add monthly calculations
                    result['Monthly_Trips'] = trips
                    result['Monthly_Carrier_Cost'] = result['Recommended_Carrier'] * trips
                    result['Monthly_Revenue'] = result['Shipper_Rate'] * trips
                    result['Monthly_Margin'] = result['Monthly_Revenue'] - result['Monthly_Carrier_Cost']
                    
                    results.append(result)
                    
                    progress_bar.progress((idx + 1) / len(routes_df))
                
                status_text.text("✅ Complete!")
                
                # Create results DataFrame
                results_df = pd.DataFrame(results)
                
                # Reorder columns
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
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Pricing Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Routes", len(results_df))
                with col2:
                    st.metric("With History", (results_df['Hist_Count'] > 0).sum())
                with col3:
                    st.metric("With Recent Data", (results_df[f'Recent_{RECENCY_CUTOFF_DAYS}d_Count'] > 0).sum())
                with col4:
                    st.metric("Total Monthly Margin", f"{results_df['Monthly_Margin'].sum():,.0f} SAR")
                
                # Show results table
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Download button
                csv_output = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv_output,
                    file_name="pricing_results.csv",
                    mime="text/csv",
                    type="primary"
                )
                
                # Show unmatched cities
                if len(unmatched_cities) > 0:
                    st.markdown("---")
                    st.subheader("⚠️ Unmatched Cities")
                    st.warning(f"The following {len(unmatched_cities)} city names could not be matched to our database. Please verify spelling.")
                    
                    unmatched_df = pd.DataFrame(unmatched_cities)
                    st.dataframe(unmatched_df, use_container_width=True, hide_index=True)
                    
                    # Add to results CSV as separate sheet
                    st.caption("These rows were still processed using the normalized/original name, but may have limited or no historical data.")
                
                # Summary by route
                with st.expander("📈 Summary Statistics"):
                    st.markdown("**By Recommendation Source:**")
                    source_summary = results_df.groupby('Recommendation_Source').agg({
                        'From': 'count',
                        'Monthly_Carrier_Cost': 'sum',
                        'Monthly_Margin': 'sum'
                    }).rename(columns={'From': 'Routes'})
                    st.dataframe(source_summary, use_container_width=True)
                    
                    st.markdown("**Routes Without Recent Data:**")
                    no_recent = results_df[results_df[f'Recent_{RECENCY_CUTOFF_DAYS}d_Count'] == 0][['From', 'To', 'Vehicle_Type', 'Hist_Count', 'Recommendation_Source']]
                    if len(no_recent) > 0:
                        st.dataframe(no_recent, use_container_width=True, hide_index=True)
                    else:
                        st.success("All routes have recent data!")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.caption("Please ensure your CSV has 'From' and 'To' columns")

# Footer
st.markdown("---")
st.caption("Built with Streamlit & CatBoost | Default: Flatbed Trailer | All prices in SAR")
