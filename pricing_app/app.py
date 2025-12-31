import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pickle
import os

st.set_page_config(page_title="Freight Pricing Tool", page_icon="🚚", layout="wide")

# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_models():
    MODEL_DIR = 'model_export'
    
    carrier_model = CatBoostRegressor()
    carrier_model.load_model(f'{MODEL_DIR}/carrier_model.cbm')
    
    shipper_model = CatBoostRegressor()
    shipper_model.load_model(f'{MODEL_DIR}/shipper_model.cbm')
    
    with open(f'{MODEL_DIR}/knn_bundle.pkl', 'rb') as f:
        knn_bundle = pickle.load(f)
    
    with open(f'{MODEL_DIR}/config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    df_knn = pd.read_parquet(f'{MODEL_DIR}/reference_data.parquet')
    
    return {
        'carrier_model': carrier_model,
        'shipper_model': shipper_model,
        'knn_model': knn_bundle['knn_model'],
        'scaler': knn_bundle['scaler'],
        'label_encoders': knn_bundle['label_encoders'],
        'config': config,
        'df_knn': df_knn
    }

models = load_models()
config = models['config']
df_knn = models['df_knn']

# Extract config
FEATURES = config['FEATURES']
CAT_FEATURES = config['CAT_FEATURES']
KNN_FEATURES = config['KNN_FEATURES']
ENTITY_MAPPING_VALUES = config.get('ENTITY_MAPPING_VALUES', ['Domestic', 'Cross_borders', 'Dry_bulk', 'Ports'])
RECENCY_CUTOFF_DAYS = config['RECENCY_CUTOFF_DAYS']
RECENCY_PENALTY_WEIGHT = config['RECENCY_PENALTY_WEIGHT']
KNN_BLEND_THRESHOLD = config.get('KNN_BLEND_THRESHOLD', 30)
DISTANCE_LOOKUP = config['DISTANCE_LOOKUP']

# ============================================================================
# UI
# ============================================================================
st.title("🚚 Freight Pricing Negotiation Tool")
st.markdown("Generate carrier price corridors with ML-powered predictions")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    anchor_discount = st.slider("Anchor Discount %", 5, 20, 12)
    ceiling_premium = st.slider("Ceiling Premium %", 3, 15, 5)
    n_similar = st.slider("Similar Loads to Show", 3, 10, 5)
    min_days_apart = st.slider("Min Days Apart", 1, 14, 7)

# Main inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Required Inputs")
    
    entity_mapping = st.selectbox(
        "Entity Mapping ⭐",
        options=ENTITY_MAPPING_VALUES,
        help="Business segment"
    )
    
    pickup_cities = sorted(df_knn['pickup_city'].unique())
    dest_cities = sorted(df_knn['destination_city'].unique())
    vehicle_types = sorted(df_knn['vehicle_type'].unique())
    
    pickup_city = st.selectbox("Pickup City", options=pickup_cities)
    destination_city = st.selectbox("Destination City", options=dest_cities)
    vehicle_type = st.selectbox("Vehicle Type", options=vehicle_types)

with col2:
    st.subheader("📦 Optional Inputs")
    st.caption("Leave empty for auto-detection")
    
    # Container - optional with auto-detect
    container_options = ['(Auto-detect)', '📦 Yes (1)', '📭 No (0)']
    container_select = st.selectbox("Container", options=container_options)
    if container_select == '(Auto-detect)':
        container = None
    elif container_select == '📦 Yes (1)':
        container = 1
    else:
        container = 0
    
    commodities = ['(Auto-detect)'] + sorted(df_knn['commodity'].unique().tolist())
    commodity_select = st.selectbox("Commodity", options=commodities)
    commodity = None if commodity_select == '(Auto-detect)' else commodity_select
    
    weight = st.number_input("Weight (Tons)", min_value=0.0, max_value=100.0, value=0.0)
    if weight == 0:
        weight = None

# ============================================================================
# PREDICTION
# ============================================================================
if st.button("🎯 Generate Negotiation Corridor", type="primary"):
    lane = f"{pickup_city} → {destination_city}"
    defaults_used = {}
    
    # Auto-detect container
    if container is None:
        em_lane_data = df_knn[(df_knn['entity_mapping'] == entity_mapping) & (df_knn['lane'] == lane)]
        if len(em_lane_data) > 0:
            container = int(em_lane_data['container'].mode().iloc[0])
        else:
            em_data = df_knn[df_knn['entity_mapping'] == entity_mapping]
            container = int(em_data['container'].mode().iloc[0]) if len(em_data) > 0 else 1
        defaults_used['container'] = 'Auto-detected'
    
    # Auto-detect commodity
    if commodity is None:
        em_lane_data = df_knn[
            (df_knn['entity_mapping'] == entity_mapping) & 
            (df_knn['lane'] == lane) &
            (df_knn['container'] == container)
        ]
        if len(em_lane_data) > 0:
            commodity = em_lane_data['commodity'].mode().iloc[0]
        else:
            em_data = df_knn[(df_knn['entity_mapping'] == entity_mapping) & (df_knn['container'] == container)]
            commodity = em_data['commodity'].mode().iloc[0] if len(em_data) > 0 else df_knn['commodity'].mode().iloc[0]
        defaults_used['commodity'] = 'Auto-detected'
    
    # Auto-detect weight
    if weight is None:
        comm_weights = df_knn[(df_knn['commodity'] == commodity) & (df_knn['weight'] > 0)]['weight']
        weight = comm_weights.median() if len(comm_weights) > 0 else df_knn['weight'].median()
        defaults_used['weight'] = 'Auto-detected'
    
    distance = DISTANCE_LOOKUP.get(lane, np.median(list(DISTANCE_LOOKUP.values())))
    
    # Model prediction
    input_data = pd.DataFrame([{
        'entity_mapping': entity_mapping,
        'commodity': commodity,
        'vehicle_type': vehicle_type,
        'pickup_city': pickup_city,
        'destination_city': destination_city,
        'distance': distance,
        'weight': weight,
        'container': container
    }])
    
    predicted_carrier = models['carrier_model'].predict(input_data[FEATURES])[0]
    predicted_shipper = models['shipper_model'].predict(input_data[FEATURES])[0]
    
    # KNN estimate
    em_lane_recent = df_knn[
        (df_knn['entity_mapping'] == entity_mapping) &
        (df_knn['lane'] == lane) & 
        (df_knn['container'] == container) &
        (df_knn['days_ago'] <= 180)
    ]
    if len(em_lane_recent) >= 2:
        weights = np.exp(-em_lane_recent['days_ago'] / 90)
        knn_estimate = np.average(em_lane_recent['total_carrier_price'], weights=weights)
        knn_source = f'Lane avg ({len(em_lane_recent)} loads)'
    else:
        knn_estimate = predicted_carrier
        knn_source = 'Model fallback'
    
    # Blend if divergent
    divergence = abs(predicted_carrier - knn_estimate) / knn_estimate * 100 if knn_estimate > 0 else 0
    if divergence > KNN_BLEND_THRESHOLD:
        blended = 0.7 * knn_estimate + 0.3 * predicted_carrier
        estimate_note = f"Blended ({divergence:.0f}% divergence)"
    else:
        blended = predicted_carrier
        estimate_note = "Model"
    
    # Corridor
    anchor_price = blended * (1 - anchor_discount / 100)
    target_price = blended
    ceiling_price = blended * (1 + ceiling_premium / 100)
    
    # Display results
    st.markdown("---")
    st.header("🎯 Negotiation Corridor")
    
    em_colors = {'Domestic': '🏠', 'Cross_borders': '🌍', 'Dry_bulk': '📦', 'Ports': '⚓'}
    cont_icon = "📦" if container == 1 else "📭"
    st.info(f"{em_colors.get(entity_mapping, '📋')} **{entity_mapping}** | {lane} | {cont_icon} Container={'Yes' if container else 'No'}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 ANCHOR", f"{anchor_price:,.0f} SAR")
    with col2:
        st.metric("🟡 TARGET", f"{target_price:,.0f} SAR")
    with col3:
        st.metric("🔴 CEILING", f"{ceiling_price:,.0f} SAR")
    
    margin = predicted_shipper - target_price
    margin_pct = margin / predicted_shipper * 100 if predicted_shipper > 0 else 0
    
    st.markdown(f"""
    **Shipper Rate:** {predicted_shipper:,.0f} SAR | **Margin:** {margin:,.0f} SAR ({margin_pct:.1f}%) | **Source:** {estimate_note}
    """)
    
    if defaults_used:
        st.caption(f"Auto-detected: {', '.join(defaults_used.keys())}")
    
    # Tiered Similar Loads
    st.markdown("---")
    st.subheader(f"🚚 Your Ammunition")
    st.caption("**Match:** ✅ EXACT (all inputs) | 🔶 LANE (same lane) | 🔄 SIMILAR (KNN)")
    
    cols = ['pickup_date', 'entity_mapping', 'commodity', 'vehicle_type', 'pickup_city', 'destination_city',
            'distance', 'weight', 'total_carrier_price', 'total_shipper_price', 'days_ago', 'lane',
            'container', 'leg_name', 'number_of_addresses']
    
    # TIER 1: EXACT
    exact_matches = df_knn[
        (df_knn['entity_mapping'] == entity_mapping) &
        (df_knn['lane'] == lane) &
        (df_knn['commodity'] == commodity) &
        (df_knn['vehicle_type'] == vehicle_type) &
        (df_knn['container'] == container)
    ].sort_values('days_ago')[cols].copy()
    exact_matches['match_type'] = 'EXACT'
    
    # TIER 2: LANE
    lane_matches = df_knn[
        (df_knn['entity_mapping'] == entity_mapping) &
        (df_knn['lane'] == lane) &
        (df_knn['container'] == container) &
        ~((df_knn['commodity'] == commodity) & (df_knn['vehicle_type'] == vehicle_type))
    ].sort_values('days_ago')[cols].copy()
    lane_matches['match_type'] = 'LANE'
    
    # TIER 3: SIMILAR
    other_lanes = df_knn[
        (df_knn['entity_mapping'] == entity_mapping) & 
        (df_knn['container'] == container) &
        (df_knn['lane'] != lane)
    ]
    if len(other_lanes) > 0:
        input_knn = []
        for col in CAT_FEATURES:
            try:
                encoded = models['label_encoders'][col].transform([input_data[col].iloc[0]])[0]
            except:
                encoded = -1
            input_knn.append(encoded)
        input_knn.extend([distance, weight, container, 0])
        
        input_scaled = models['scaler'].transform([input_knn])
        input_scaled[0, -1] *= RECENCY_PENALTY_WEIGHT
        
        X_other = models['scaler'].transform(other_lanes[KNN_FEATURES])
        X_other[:, -1] *= RECENCY_PENALTY_WEIGHT
        
        knn_temp = NearestNeighbors(n_neighbors=min(n_similar * 3, len(other_lanes)))
        knn_temp.fit(X_other)
        _, indices = knn_temp.kneighbors(input_scaled)
        similar_matches = other_lanes.iloc[indices[0]][cols].copy()
        similar_matches['match_type'] = 'SIMILAR'
    else:
        similar_matches = pd.DataFrame()
    
    # Combine with time spread
    combined = pd.concat([exact_matches, lane_matches, similar_matches], ignore_index=True)
    
    result_indices, result_days = [], []
    for tier in ['EXACT', 'LANE', 'SIMILAR']:
        tier_data = combined[combined['match_type'] == tier]
        for idx, row in tier_data.iterrows():
            days = row['days_ago']
            if len(result_days) == 0 or all(abs(days - d) >= min_days_apart for d in result_days):
                result_indices.append(idx)
                result_days.append(days)
            if len(result_indices) >= n_similar:
                break
        if len(result_indices) >= n_similar:
            break
    
    if len(result_indices) < n_similar:
        for idx in combined.index:
            if idx not in result_indices:
                result_indices.append(idx)
            if len(result_indices) >= n_similar:
                break
    
    if result_indices:
        result = combined.loc[result_indices].copy()
        result['margin'] = result['total_shipper_price'] - result['total_carrier_price']
        result['is_recent'] = result['days_ago'] <= RECENCY_CUTOFF_DAYS
        
        def get_match_label(row):
            mt = row['match_type']
            recent = '🔥' if row['is_recent'] else ''
            cont = '📦' if row['container'] == 1 else ''
            if mt == 'EXACT':
                return f'✅ EXACT {recent} {cont}'
            elif mt == 'LANE':
                return f'🔶 LANE {recent} {cont}'
            else:
                return f'🔄 SIMILAR {recent} {cont}'
        
        result['Match'] = result.apply(get_match_label, axis=1)
        
        display_cols = ['Match', 'pickup_date', 'commodity', 'vehicle_type', 'weight', 'total_carrier_price', 'margin', 'days_ago']
        display_df = result[display_cols].copy()
        display_df.columns = ['Match', 'Date', 'Commodity', 'Vehicle', 'Weight', 'Carrier', 'Margin', 'Days']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        exact_count = (result['match_type'] == 'EXACT').sum()
        lane_count = (result['match_type'] == 'LANE').sum()
        similar_count = (result['match_type'] == 'SIMILAR').sum()
        st.caption(f"**Summary:** EXACT={exact_count} | LANE={lane_count} | SIMILAR={similar_count}")
    else:
        st.warning("No matches found")

st.markdown("---")
st.caption("Match Priority: EXACT > LANE > SIMILAR | 📦=Container | 🔥=Recent")
