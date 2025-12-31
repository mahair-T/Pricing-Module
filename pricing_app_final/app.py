import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.neighbors import NearestNeighbors
import pickle
import os

st.set_page_config(page_title="Freight Pricing Tool", page_icon="🚚", layout="wide")

# Get the directory where app.py is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))

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
    
    with open(os.path.join(MODEL_DIR, 'knn_bundle.pkl'), 'rb') as f:
        knn_bundle = pickle.load(f)
    
    with open(os.path.join(MODEL_DIR, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)
    
    df_knn = pd.read_parquet(os.path.join(MODEL_DIR, 'reference_data.parquet'))
    
    return {
        'carrier_model': carrier_model, 'shipper_model': shipper_model,
        'scaler': knn_bundle['scaler'], 'label_encoders': knn_bundle['label_encoders'],
        'config': config, 'df_knn': df_knn
    }

models = load_models()
config = models['config']
df_knn = models['df_knn']

FEATURES = config['FEATURES']
CAT_FEATURES = config['CAT_FEATURES']
KNN_FEATURES = config['KNN_FEATURES']
ENTITY_MAPPING_VALUES = config.get('ENTITY_MAPPING_VALUES', ['Domestic', 'Cross_borders', 'Dry_bulk', 'Ports'])
RECENCY_CUTOFF_DAYS = config.get('RECENCY_CUTOFF_DAYS', 90)
KNN_BLEND_THRESHOLD = config.get('KNN_BLEND_THRESHOLD', 30)
DISTANCE_LOOKUP = config['DISTANCE_LOOKUP']
BACKHAUL_LOOKUP = config.get('BACKHAUL_LOOKUP', {})
LANE_STATS = config.get('LANE_STATS', {})
RARE_LANE_THRESHOLD = config.get('RARE_LANE_THRESHOLD', 10)

st.title("🚚 Freight Pricing Negotiation Tool")
st.caption("ML-powered pricing • Vehicle type required • 90-day recency priority")

with st.sidebar:
    st.header("⚙️ Settings")
    anchor_discount = st.slider("Anchor Discount %", 5, 20, 12)
    ceiling_premium = st.slider("Ceiling Premium %", 3, 15, 5)
    n_similar = st.slider("Similar Loads", 3, 10, 5)
    min_days_apart = st.slider("Min Days Apart", 1, 14, 7)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Required Inputs")
    entity_mapping = st.selectbox("Entity Mapping ⭐", options=ENTITY_MAPPING_VALUES)
    pickup_city = st.selectbox("Pickup City", options=sorted(df_knn['pickup_city'].unique()))
    destination_city = st.selectbox("Destination City", options=sorted(df_knn['destination_city'].unique()))
    vehicle_type = st.selectbox("Vehicle Type ⭐", options=sorted(df_knn['vehicle_type'].unique()))

with col2:
    st.subheader("📦 Optional Inputs")
    container_select = st.selectbox("Container", ['(Auto-detect)', '📦 Yes (Containerized)', '📭 No'])
    container = None if container_select == '(Auto-detect)' else (1 if 'Yes' in container_select else 0)
    
    commodities = ['(Auto-detect)'] + sorted(df_knn['commodity'].unique().tolist())
    commodity_select = st.selectbox("Commodity", options=commodities)
    commodity = None if commodity_select == '(Auto-detect)' else commodity_select
    
    weight = st.number_input("Weight (Tons)", min_value=0.0, max_value=100.0, value=0.0)
    weight = None if weight == 0 else weight

if st.button("🎯 Generate Negotiation Corridor", type="primary"):
    lane = f"{pickup_city} → {destination_city}"
    defaults_used = {}
    
    # Lane stats
    lane_stats = LANE_STATS.get(lane, {})
    lane_load_count = lane_stats.get('load_count', 0)
    if pd.isna(lane_load_count):
        lane_load_count = 0
    dest_backhaul_prob = BACKHAUL_LOOKUP.get(destination_city, 0)
    is_rare_lane = lane_load_count < RARE_LANE_THRESHOLD
    
    # Auto-detect container
    if container is None:
        em_lane_data = df_knn[(df_knn['entity_mapping'] == entity_mapping) & (df_knn['lane'] == lane) & (df_knn['vehicle_type'] == vehicle_type)]
        if len(em_lane_data) > 0:
            container = int(em_lane_data['container'].mode().iloc[0])
        else:
            em_data = df_knn[(df_knn['entity_mapping'] == entity_mapping) & (df_knn['vehicle_type'] == vehicle_type)]
            container = int(em_data['container'].mode().iloc[0]) if len(em_data) > 0 else 1
        defaults_used['container'] = True
    
    # Auto-detect commodity
    if commodity is None:
        em_lane_data = df_knn[(df_knn['entity_mapping'] == entity_mapping) & (df_knn['lane'] == lane) & 
                              (df_knn['vehicle_type'] == vehicle_type) & (df_knn['container'] == container)]
        if len(em_lane_data) > 0:
            commodity = em_lane_data['commodity'].mode().iloc[0]
        else:
            em_data = df_knn[(df_knn['entity_mapping'] == entity_mapping) & (df_knn['vehicle_type'] == vehicle_type)]
            commodity = em_data['commodity'].mode().iloc[0] if len(em_data) > 0 else df_knn['commodity'].mode().iloc[0]
        defaults_used['commodity'] = True
    
    # Auto-detect weight
    if weight is None:
        comm_weights = df_knn[(df_knn['commodity'] == commodity) & (df_knn['weight'] > 0)]['weight']
        weight = comm_weights.median() if len(comm_weights) > 0 else df_knn['weight'].median()
        defaults_used['weight'] = True
    
    # Get distance
    distance = DISTANCE_LOOKUP.get(lane, np.median(list(DISTANCE_LOOKUP.values())))
    
    # Auto-detect is_multistop
    is_multistop = 0
    if pickup_city == destination_city and distance > 100:
        is_multistop = 1
    
    # Model prediction
    input_data = pd.DataFrame([{
        'entity_mapping': entity_mapping,
        'commodity': commodity,
        'vehicle_type': vehicle_type,
        'pickup_city': pickup_city,
        'destination_city': destination_city,
        'distance': distance,
        'weight': weight,
        'container': container,
        'is_multistop': is_multistop
    }])
    
    predicted_carrier = models['carrier_model'].predict(input_data[FEATURES])[0]
    predicted_shipper = models['shipper_model'].predict(input_data[FEATURES])[0]
    
    # Rare lane handling
    rare_lane_info = None
    if is_rare_lane:
        bh_low = max(0, dest_backhaul_prob - 15)
        bh_high = min(100, dest_backhaul_prob + 15)
        
        similar_bh = df_knn[
            (df_knn['entity_mapping'] == entity_mapping) &
            (df_knn['vehicle_type'] == vehicle_type) &  # Vehicle type required
            (df_knn['dest_backhaul_prob'] >= bh_low) &
            (df_knn['dest_backhaul_prob'] <= bh_high) &
            (df_knn['distance'] > 50) &
            (df_knn['pickup_city'] != df_knn['destination_city'])
        ].copy()
        
        if len(similar_bh) > 0:
            similar_bh['carrier_per_km'] = similar_bh['total_carrier_price'] / similar_bh['distance']
            gb_per_km = similar_bh['carrier_per_km'].median()
            backhaul_estimate = gb_per_km * distance
            rare_lane_info = {
                'count': len(similar_bh),
                'bracket': f'{bh_low:.0f}-{bh_high:.0f}%',
                'gb_per_km': round(gb_per_km, 2),
                'estimate': round(backhaul_estimate, 0)
            }
    
    # KNN estimate - 90 day cutoff with vehicle type required
    em_lane_recent = df_knn[
        (df_knn['entity_mapping'] == entity_mapping) &
        (df_knn['lane'] == lane) &
        (df_knn['vehicle_type'] == vehicle_type) &  # Vehicle type required
        (df_knn['container'] == container) &
        (df_knn['is_multistop'] == is_multistop) &
        (df_knn['days_ago'] <= RECENCY_CUTOFF_DAYS)  # 90-day cutoff
    ]
    
    if len(em_lane_recent) >= 2:
        # Weight by recency (exponential decay, half-life ~30 days)
        weights = np.exp(-em_lane_recent['days_ago'] / 30)
        knn_estimate = np.average(em_lane_recent['total_carrier_price'], weights=weights)
        knn_source = f'Recent lane avg ({len(em_lane_recent)} loads, ≤{RECENCY_CUTOFF_DAYS}d)'
    else:
        knn_estimate = predicted_carrier
        knn_source = 'Model (no recent matches)'
    
    # Blend
    if is_rare_lane and rare_lane_info:
        blended = 0.5 * predicted_carrier + 0.5 * rare_lane_info['estimate']
        estimate_source = 'Rare lane blend'
    elif knn_estimate != predicted_carrier:
        divergence = abs(predicted_carrier - knn_estimate) / knn_estimate * 100
        if divergence > KNN_BLEND_THRESHOLD:
            blended = 0.7 * knn_estimate + 0.3 * predicted_carrier
            estimate_source = f'Blended ({divergence:.0f}% div)'
        else:
            blended = predicted_carrier
            estimate_source = 'Model'
    else:
        blended = predicted_carrier
        estimate_source = 'Model'
    
    # Corridor
    anchor_price = blended * (1 - anchor_discount / 100)
    target_price = blended
    ceiling_price = blended * (1 + ceiling_premium / 100)
    
    # Display results
    st.markdown("---")
    st.header("🎯 Negotiation Corridor")
    
    if is_rare_lane:
        st.warning(f"⚠️ **RARE LANE** - Only {int(lane_load_count)} historical loads")
        if rare_lane_info:
            st.caption(f"Using backhaul-based pricing: {rare_lane_info['count']} similar lanes (BH {rare_lane_info['bracket']}), GB/km={rare_lane_info['gb_per_km']}")
    
    em_icons = {'Domestic': '🏠', 'Cross_borders': '🌍', 'Dry_bulk': '📦', 'Ports': '⚓'}
    cont_icon = '📦' if container == 1 else '📭'
    ms_icon = '🔄' if is_multistop == 1 else '➡️'
    st.info(f"{em_icons.get(entity_mapping, '📋')} **{entity_mapping}** | {lane} | 🚛 {vehicle_type} | {cont_icon} | {ms_icon}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 ANCHOR (Start)", f"{anchor_price:,.0f} SAR")
    with col2:
        st.metric("🟡 TARGET (Fair)", f"{target_price:,.0f} SAR")
    with col3:
        st.metric("🔴 CEILING (Max)", f"{ceiling_price:,.0f} SAR")
    
    margin = predicted_shipper - target_price
    margin_pct = margin / predicted_shipper * 100 if predicted_shipper > 0 else 0
    st.markdown(f"**Shipper Rate:** {predicted_shipper:,.0f} SAR | **Margin:** {margin:,.0f} SAR ({margin_pct:.1f}%) | **Source:** {estimate_source}")
    
    if defaults_used:
        st.caption(f"Auto-detected: {', '.join(defaults_used.keys())}")
    
    # ==========================================
    # SIMILAR LOADS - NEW LOGIC
    # Vehicle type REQUIRED, 90-day priority
    # ==========================================
    st.markdown("---")
    st.subheader("🚚 Your Ammunition")
    st.caption(f"**Vehicle type required** • **≤{RECENCY_CUTOFF_DAYS} days priority** • ✅ EXACT | 🔶 LANE | 🔄 SIMILAR")
    
    cols = ['pickup_date', 'entity_mapping', 'commodity', 'vehicle_type', 'pickup_city', 'destination_city',
            'distance', 'weight', 'total_carrier_price', 'total_shipper_price', 'days_ago', 'lane',
            'container', 'is_multistop', 'dest_backhaul_prob']
    
    # Base filter: entity + vehicle_type (REQUIRED)
    base_filter = (df_knn['entity_mapping'] == entity_mapping) & (df_knn['vehicle_type'] == vehicle_type)
    
    # Recent filter (≤90 days)
    recent_filter = base_filter & (df_knn['days_ago'] <= RECENCY_CUTOFF_DAYS)
    
    # TIER 1: EXACT - same lane + commodity + container + multistop (recent only)
    exact_recent = df_knn[
        recent_filter & 
        (df_knn['lane'] == lane) &
        (df_knn['commodity'] == commodity) &
        (df_knn['container'] == container) &
        (df_knn['is_multistop'] == is_multistop)
    ].sort_values('days_ago')[cols].copy()
    exact_recent['match_type'] = 'EXACT'
    
    # TIER 2: LANE - same lane + container + multistop, different commodity (recent only)
    lane_recent = df_knn[
        recent_filter &
        (df_knn['lane'] == lane) &
        (df_knn['container'] == container) &
        (df_knn['is_multistop'] == is_multistop) &
        (df_knn['commodity'] != commodity)
    ].sort_values('days_ago')[cols].copy()
    lane_recent['match_type'] = 'LANE'
    
    # TIER 3: SIMILAR - same container, different lane, similar distance (recent only)
    dist_tol = distance * 0.25
    similar_recent = df_knn[
        recent_filter &
        (df_knn['container'] == container) &
        (df_knn['lane'] != lane) &
        (df_knn['distance'] >= distance - dist_tol) &
        (df_knn['distance'] <= distance + dist_tol)
    ].sort_values('days_ago').head(n_similar * 3)[cols].copy()
    similar_recent['match_type'] = 'SIMILAR'
    
    # Combine recent matches
    combined = pd.concat([exact_recent, lane_recent, similar_recent], ignore_index=True)
    
    # If no recent matches at all, fall back to all-time (with warning)
    used_fallback = False
    if len(combined) == 0:
        used_fallback = True
        # Fallback: same filters but no time restriction
        exact_all = df_knn[
            base_filter & 
            (df_knn['lane'] == lane) &
            (df_knn['commodity'] == commodity) &
            (df_knn['container'] == container) &
            (df_knn['is_multistop'] == is_multistop)
        ].sort_values('days_ago')[cols].copy()
        exact_all['match_type'] = 'EXACT'
        
        lane_all = df_knn[
            base_filter &
            (df_knn['lane'] == lane) &
            (df_knn['container'] == container) &
            (df_knn['is_multistop'] == is_multistop) &
            (df_knn['commodity'] != commodity)
        ].sort_values('days_ago')[cols].copy()
        lane_all['match_type'] = 'LANE'
        
        similar_all = df_knn[
            base_filter &
            (df_knn['container'] == container) &
            (df_knn['lane'] != lane) &
            (df_knn['distance'] >= distance - dist_tol) &
            (df_knn['distance'] <= distance + dist_tol)
        ].sort_values('days_ago').head(n_similar * 3)[cols].copy()
        similar_all['match_type'] = 'SIMILAR'
        
        combined = pd.concat([exact_all, lane_all, similar_all], ignore_index=True)
    
    # Select with time spread (prioritize most recent)
    result_idx, result_days = [], []
    for tier in ['EXACT', 'LANE', 'SIMILAR']:
        tier_data = combined[combined['match_type'] == tier].sort_values('days_ago')  # Most recent first
        for idx, row in tier_data.iterrows():
            days = row['days_ago']
            if len(result_days) == 0 or all(abs(days - d) >= min_days_apart for d in result_days):
                result_idx.append(idx)
                result_days.append(days)
            if len(result_idx) >= n_similar:
                break
        if len(result_idx) >= n_similar:
            break
    
    # Fill remaining if needed
    if len(result_idx) < n_similar:
        for idx in combined.sort_values('days_ago').index:  # Most recent first
            if idx not in result_idx:
                result_idx.append(idx)
            if len(result_idx) >= n_similar:
                break
    
    if used_fallback and len(combined) > 0:
        st.warning(f"⚠️ No matches within {RECENCY_CUTOFF_DAYS} days - showing older data")
    
    if result_idx:
        result = combined.loc[result_idx].copy()
        result['margin'] = result['total_shipper_price'] - result['total_carrier_price']
        result['is_recent'] = result['days_ago'] <= RECENCY_CUTOFF_DAYS
        
        def get_label(row):
            mt = '✅' if row['match_type'] == 'EXACT' else ('🔶' if row['match_type'] == 'LANE' else '🔄')
            r = '🔥' if row['is_recent'] else '⏳'  # ⏳ for old data
            c = '📦' if row['container'] == 1 else ''
            ms = '🔄' if row['is_multistop'] == 1 else ''
            return f"{mt} {r} {c} {ms}".strip()
        
        result['Match'] = result.apply(get_label, axis=1)
        
        # Sort by days_ago (most recent first)
        result = result.sort_values('days_ago')
        
        display_df = result[['Match', 'pickup_date', 'lane', 'commodity', 'weight', 'total_carrier_price', 'margin', 'days_ago']]
        display_df.columns = ['Match', 'Date', 'Lane', 'Commodity', 'Weight', 'Carrier', 'Margin', 'Days']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        exact_n = (result['match_type'] == 'EXACT').sum()
        lane_n = (result['match_type'] == 'LANE').sum()
        similar_n = (result['match_type'] == 'SIMILAR').sum()
        recent_n = result['is_recent'].sum()
        st.caption(f"**Summary:** EXACT={exact_n} | LANE={lane_n} | SIMILAR={similar_n} | Recent={recent_n}/{len(result)}")
    else:
        st.warning("No similar loads found for this vehicle type")

st.markdown("---")
st.caption(f"✅=EXACT | 🔶=LANE | 🔄=SIMILAR | 🔥=Recent (≤{RECENCY_CUTOFF_DAYS}d) | ⏳=Older | 📦=Container")
