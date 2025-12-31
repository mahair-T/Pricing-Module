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
    
    # Try JSON format first (more portable), fallback to CBM
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
RECENCY_CUTOFF_DAYS = config['RECENCY_CUTOFF_DAYS']
KNN_BLEND_THRESHOLD = config.get('KNN_BLEND_THRESHOLD', 30)
DISTANCE_LOOKUP = config['DISTANCE_LOOKUP']
BACKHAUL_LOOKUP = config.get('BACKHAUL_LOOKUP', {})
LANE_STATS = config.get('LANE_STATS', {})
RARE_LANE_THRESHOLD = config.get('RARE_LANE_THRESHOLD', 10)

st.title("🚚 Freight Pricing Negotiation Tool")
st.caption("ML-powered pricing with container, multi-stop, and rare lane support")

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
    vehicle_type = st.selectbox("Vehicle Type", options=sorted(df_knn['vehicle_type'].unique()))

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
    
    # Lane stats for rare lane detection
    lane_stats = LANE_STATS.get(lane, {})
    lane_load_count = lane_stats.get('load_count', 0)
    if pd.isna(lane_load_count):
        lane_load_count = 0
    dest_backhaul_prob = BACKHAUL_LOOKUP.get(destination_city, 0)
    is_rare_lane = lane_load_count < RARE_LANE_THRESHOLD
    
    # Auto-detect container
    if container is None:
        em_lane_data = df_knn[(df_knn['entity_mapping'] == entity_mapping) & (df_knn['lane'] == lane)]
        if len(em_lane_data) > 0:
            container = int(em_lane_data['container'].mode().iloc[0])
        else:
            em_data = df_knn[df_knn['entity_mapping'] == entity_mapping]
            container = int(em_data['container'].mode().iloc[0]) if len(em_data) > 0 else 1
        defaults_used['container'] = True
    
    # Auto-detect commodity
    if commodity is None:
        em_lane_data = df_knn[(df_knn['entity_mapping'] == entity_mapping) & (df_knn['lane'] == lane) & (df_knn['container'] == container)]
        if len(em_lane_data) > 0:
            commodity = em_lane_data['commodity'].mode().iloc[0]
        else:
            em_data = df_knn[(df_knn['entity_mapping'] == entity_mapping) & (df_knn['container'] == container)]
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
    
    # Rare lane handling (backhaul-based pricing)
    rare_lane_info = None
    if is_rare_lane:
        bh_low = max(0, dest_backhaul_prob - 15)
        bh_high = min(100, dest_backhaul_prob + 15)
        
        similar_bh = df_knn[
            (df_knn['entity_mapping'] == entity_mapping) &
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
    
    # KNN estimate for blending
    em_lane_recent = df_knn[
        (df_knn['entity_mapping'] == entity_mapping) &
        (df_knn['lane'] == lane) &
        (df_knn['container'] == container) &
        (df_knn['is_multistop'] == is_multistop) &
        (df_knn['days_ago'] <= 180)
    ]
    if len(em_lane_recent) >= 2:
        weights = np.exp(-em_lane_recent['days_ago'] / 90)
        knn_estimate = np.average(em_lane_recent['total_carrier_price'], weights=weights)
        knn_source = f'Lane avg ({len(em_lane_recent)} loads)'
    else:
        knn_estimate = predicted_carrier
        knn_source = 'Model fallback'
    
    # Blend / select best estimate
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
    
    # Rare lane warning
    if is_rare_lane:
        st.warning(f"⚠️ **RARE LANE** - Only {int(lane_load_count)} historical loads")
        if rare_lane_info:
            st.caption(f"Using backhaul-based pricing: {rare_lane_info['count']} similar lanes (BH {rare_lane_info['bracket']}), GB/km={rare_lane_info['gb_per_km']}")
    
    # Info bar
    em_icons = {'Domestic': '🏠', 'Cross_borders': '🌍', 'Dry_bulk': '📦', 'Ports': '⚓'}
    cont_icon = '📦' if container == 1 else '📭'
    ms_icon = '🔄' if is_multistop == 1 else '➡️'
    st.info(f"{em_icons.get(entity_mapping, '📋')} **{entity_mapping}** | {lane} | {cont_icon} Container | {ms_icon} {'Multi-stop' if is_multistop else 'Direct'}")
    
    # Corridor metrics
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
    
    # Similar loads
    st.markdown("---")
    st.subheader("🚚 Your Ammunition")
    st.caption("**Match:** ✅ EXACT | 🔶 LANE | 🔄 SIMILAR")
    
    cols = ['pickup_date', 'entity_mapping', 'commodity', 'vehicle_type', 'pickup_city', 'destination_city',
            'distance', 'weight', 'total_carrier_price', 'total_shipper_price', 'days_ago', 'lane',
            'container', 'is_multistop', 'dest_backhaul_prob']
    
    # EXACT matches
    exact = df_knn[
        (df_knn['entity_mapping'] == entity_mapping) & (df_knn['lane'] == lane) &
        (df_knn['commodity'] == commodity) & (df_knn['vehicle_type'] == vehicle_type) &
        (df_knn['container'] == container) & (df_knn['is_multistop'] == is_multistop)
    ].sort_values('days_ago')[cols].copy()
    exact['match_type'] = 'EXACT'
    
    # LANE matches
    lane_m = df_knn[
        (df_knn['entity_mapping'] == entity_mapping) & (df_knn['lane'] == lane) &
        (df_knn['container'] == container) & (df_knn['is_multistop'] == is_multistop) &
        ~((df_knn['commodity'] == commodity) & (df_knn['vehicle_type'] == vehicle_type))
    ].sort_values('days_ago')[cols].copy()
    lane_m['match_type'] = 'LANE'
    
    # SIMILAR matches (same entity + container, different lane)
    other = df_knn[
        (df_knn['entity_mapping'] == entity_mapping) &
        (df_knn['container'] == container) &
        (df_knn['lane'] != lane)
    ]
    if len(other) > 0:
        dist_tol = distance * 0.25
        similar_m = other[
            (other['distance'] >= distance - dist_tol) &
            (other['distance'] <= distance + dist_tol)
        ].sort_values('days_ago').head(n_similar * 2)[cols].copy()
        similar_m['match_type'] = 'SIMILAR'
    else:
        similar_m = pd.DataFrame()
    
    # Combine with priority
    combined = pd.concat([exact, lane_m, similar_m], ignore_index=True)
    
    # Select with time spread
    result_idx, result_days = [], []
    for tier in ['EXACT', 'LANE', 'SIMILAR']:
        tier_data = combined[combined['match_type'] == tier]
        for idx, row in tier_data.iterrows():
            days = row['days_ago']
            if len(result_days) == 0 or all(abs(days - d) >= min_days_apart for d in result_days):
                result_idx.append(idx)
                result_days.append(days)
            if len(result_idx) >= n_similar:
                break
        if len(result_idx) >= n_similar:
            break
    
    # Fill remaining
    if len(result_idx) < n_similar:
        for idx in combined.index:
            if idx not in result_idx:
                result_idx.append(idx)
            if len(result_idx) >= n_similar:
                break
    
    if result_idx:
        result = combined.loc[result_idx].copy()
        result['margin'] = result['total_shipper_price'] - result['total_carrier_price']
        result['is_recent'] = result['days_ago'] <= RECENCY_CUTOFF_DAYS
        
        def get_label(row):
            mt = '✅' if row['match_type'] == 'EXACT' else ('🔶' if row['match_type'] == 'LANE' else '🔄')
            r = '🔥' if row['is_recent'] else ''
            c = '📦' if row['container'] == 1 else ''
            ms = '🔄' if row['is_multistop'] == 1 else ''
            return f"{mt} {r} {c} {ms}".strip()
        
        result['Match'] = result.apply(get_label, axis=1)
        
        display_df = result[['Match', 'pickup_date', 'commodity', 'vehicle_type', 'weight', 'total_carrier_price', 'margin', 'days_ago']]
        display_df.columns = ['Match', 'Date', 'Commodity', 'Vehicle', 'Weight', 'Carrier', 'Margin', 'Days']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        exact_n = (result['match_type'] == 'EXACT').sum()
        lane_n = (result['match_type'] == 'LANE').sum()
        similar_n = (result['match_type'] == 'SIMILAR').sum()
        st.caption(f"**Summary:** EXACT={exact_n} | LANE={lane_n} | SIMILAR={similar_n}")
    else:
        st.warning("No similar loads found")

st.markdown("---")
st.caption("✅=EXACT | 🔶=LANE | 🔄=SIMILAR | 🔥=Recent | 📦=Container | 🔄=Multi-stop")
