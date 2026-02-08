#!/usr/bin/env python3
"""
Freight Pricing Model — Production Training Pipeline
=====================================================
Fetches fresh data from Redash, retrains domestic + port models,
and exports artifacts to model_export/.

Designed to run daily via GitHub Actions at 08:00 AST (05:00 UTC).
"""

import os
import sys
import time
import pickle
import requests
import pandas as pd
import numpy as np
import warnings
from collections import defaultdict
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from sklearn.linear_model import HuberRegressor

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("⚠️  rapidfuzz not available — fuzzy matching suggestions disabled")

warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              CONFIGURATION                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# --- Redash API endpoints ---------------------------------------------------
REDASH_DOMESTIC_URL = os.environ.get(
    "REDASH_DOMESTIC_URL",
    "https://redash.trella.co/api/queries/4869/results.csv?api_key=WKRrDCVWxzKhde4EberhP5Ce2qM7Zj80nurNZOwh",
)
REDASH_PORT_TRIP_URL = os.environ.get(
    "REDASH_PORT_TRIP_URL",
    "https://redash.trella.co/api/queries/4871/results.csv?api_key=2aKbUwmfBKoMxkP5WgRb2XONhGDSSJlR2aIspo2a",
)
REDASH_PORT_DIRECT_URL = os.environ.get(
    "REDASH_PORT_DIRECT_URL",
    "https://redash.trella.co/api/queries/4870/results.csv?api_key=92MLI3IcFu0vT5lTBW1KkxAQkYy5YIHFLbJssUk2",
)

# --- Static data (committed to repo) -----------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(REPO_ROOT, "pricing_app_final", "model_export")
DISTANCE_MATRIX_FILE = os.path.join(STATIC_DIR, "distance_matrix.pkl")
CITY_REGIONS_FILE = os.path.join(STATIC_DIR, "city_normalization_with_regions.csv")
BACKHAUL_FILE = os.path.join(STATIC_DIR, "Backhaul Prob.csv")

# --- Output -------------------------------------------------------------------
EXPORT_DIR = os.path.join(REPO_ROOT, "pricing_app_final", "model_export")

# --- Domestic data filters ----------------------------------------------------
ENTITY_MAPPING = "Domestic"
VEHICLE_TYPES = ["تريلا فرش", "تريلا ستائر"]
MIN_CARRIER_PRICE = 50
MAX_CARRIER_PRICE = 100_000
MIN_DISTANCE = 50
MIN_CPK = 0.5
MAX_CPK = 5.0

# --- Index + Shrinkage --------------------------------------------------------
HALF_LIFE_DAYS = 90
SHRINKAGE_K = 10
N_BENCHMARK_LANES = 5
INDEX_SMOOTHING_DAYS = 7

# --- Spatial model ------------------------------------------------------------
SPATIAL_RADIUS_KM = 150
SPATIAL_PERCENTILE = 50
SPATIAL_MIN_NEIGHBORS = 3
SPATIAL_RECENCY_DAYS = 90

# --- Validation ---------------------------------------------------------------
HOLDOUT_DAYS = 30
RECENCY_CUTOFF_DAYS = 90

# --- Port pricing -------------------------------------------------------------
PORT_MIN_CARRIER_PRICE = 100
PORT_MAX_CARRIER_PRICE = 50_000
PORT_MIN_DISTANCE = 10
PORT_MIN_CPK = 0.3
PORT_MAX_CPK = 12.0
PORT_DAYS_LOOKBACK = None  # None = ALL TIME
PORT_CORE_TOP_N = 11


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            HELPER FUNCTIONS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def fetch_csv(url: str, label: str, retries: int = 3, timeout: int = 120) -> pd.DataFrame:
    """Download a CSV from Redash with retries."""
    for attempt in range(1, retries + 1):
        try:
            print(f"   ↓ Fetching {label} (attempt {attempt}/{retries}) …")
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            print(f"     ✓ {len(df):,} rows")
            return df
        except Exception as e:
            print(f"     ✗ {e}")
            if attempt < retries:
                time.sleep(10 * attempt)
            else:
                raise RuntimeError(f"Failed to fetch {label} after {retries} attempts") from e


def clean_numeric_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Clean numeric columns (remove commas, coerce)."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""), errors="coerce"
            )
    return df


def validate_port_cities(df, variant_lookup, fuzzy_threshold=70):
    """Validate all true_drop_off cities can be normalised."""
    if "true_drop_off" not in df.columns:
        print("   ℹ️  No true_drop_off column — skipping validation")
        return True

    tdo_cities = df["true_drop_off"].dropna().astype(str).str.strip()
    tdo_cities = tdo_cities[tdo_cities != ""].unique()

    unmatched = []
    for city in tdo_cities:
        if city not in variant_lookup:
            suggestions = []
            if RAPIDFUZZ_AVAILABLE:
                matches = process.extract(
                    city, list(variant_lookup.keys()), scorer=fuzz.ratio, limit=3
                )
                suggestions = [(m[0], m[1]) for m in matches if m[1] >= fuzzy_threshold]
            count = int(len(df[df["true_drop_off"] == city]))
            unmatched.append({"city": city, "count": count, "suggestions": suggestions})

    if unmatched:
        print(f"\n🚨 {len(unmatched)} UNMATCHED true_drop_off cities:")
        for u in sorted(unmatched, key=lambda x: -x["count"]):
            sugg = ", ".join(f"'{s[0]}' ({s[1]}%)" for s in u["suggestions"]) or "none"
            print(f"   '{u['city']}' ({u['count']:,} rows) → suggestions: {sugg}")
        raise ValueError(
            f"{len(unmatched)} unmatched true_drop_off cities — update {CITY_REGIONS_FILE}"
        )

    print(f"   ✅ All {len(tdo_cities)} true_drop_off cities validated")
    return True


def fit_linear_transform(X, y, method="huber"):
    """Fit Port = α × Domestic + β."""
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    model = HuberRegressor(epsilon=1.35) if method == "huber" else __import__("sklearn.linear_model", fromlist=["LinearRegression"]).LinearRegression()
    model.fit(X, y)
    alpha, beta = model.coef_[0], model.intercept_
    y_pred = alpha * X.flatten() + beta
    return {
        "alpha": alpha,
        "beta": beta,
        "mae": mean_absolute_error(y, y_pred),
        "median_ae": median_absolute_error(y, y_pred),
        "r2": r2_score(y, y_pred),
        "n": len(y),
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           PREDICTION FUNCTIONS                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def predict_index_shrinkage(pickup, dest, model):
    """Predict using Index + Shrinkage (90d HL)."""
    lane = f"{pickup} → {dest}"

    # Index prediction
    idx_pred = None
    if lane in model["lane_multipliers"]:
        idx_pred = model["current_index"] * model["lane_multipliers"][lane]

    # Shrinkage prediction
    p_prior = model["pickup_priors"].get(pickup, model["global_mean"])
    d_prior = model["dest_priors"].get(dest, model["global_mean"])
    city_prior = (p_prior + d_prior) / 2

    if lane in model["lane_stats"]:
        stats = model["lane_stats"][lane]
        lam = stats["lane_n"] / (stats["lane_n"] + model["k"])
        shrink_pred = lam * stats["lane_mean"] + (1 - lam) * city_prior
        method = "Index + Shrinkage" if idx_pred else "Shrinkage"
    else:
        shrink_pred = city_prior
        method = "City Prior"

    final = (idx_pred + shrink_pred) / 2 if idx_pred is not None else shrink_pred
    return final, method


def predict_spatial(pickup, dest, model):
    """Predict using Spatial R150 P50 IDW."""
    cfg = model["config"]
    radius_km = cfg["radius_km"]
    min_neighbors = cfg["min_neighbors"]
    percentile = cfg["percentile"]

    pickup_neighbors = [
        (c, d) for c, d in model["city_neighbors"].get(pickup, []) if d <= radius_km
    ]
    dest_neighbors = [
        (c, d) for c, d in model["city_neighbors"].get(dest, []) if d <= radius_km
    ]

    pickup_data = [
        (model["city_outbound_cpk"][n]["outbound_cpk_mean"], d)
        for n, d in pickup_neighbors
        if n in model["city_outbound_cpk"]
    ]
    dest_data = [
        (model["city_inbound_cpk"][n]["inbound_cpk_mean"], d)
        for n, d in dest_neighbors
        if n in model["city_inbound_cpk"]
    ]

    if len(pickup_data) >= min_neighbors and len(dest_data) >= min_neighbors:
        epsilon = 1
        p_weights = [1 / (d + epsilon) for _, d in pickup_data]
        p_weighted = sum(cpk * w for (cpk, _), w in zip(pickup_data, p_weights)) / sum(p_weights)
        d_weights = [1 / (d + epsilon) for _, d in dest_data]
        d_weighted = sum(cpk * w for (cpk, _), w in zip(dest_data, d_weights)) / sum(d_weights)
        return (p_weighted + d_weighted) / 2, "Spatial IDW"

    p_prov = model["city_to_province"].get(pickup)
    d_prov = model["city_to_province"].get(dest)
    if p_prov and d_prov and (p_prov, d_prov) in model["province_pair_cpk"]:
        return model["province_pair_cpk"][(p_prov, d_prov)][f"p{percentile}"], f"Province P{percentile}"

    return model["global_mean"], "Global Mean"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              MAIN PIPELINE                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    start = time.time()

    # ── 1. Fetch data from Redash ─────────────────────────────────────────────
    print("=" * 70)
    print("📡 FETCHING DATA FROM REDASH")
    print("=" * 70)

    df_raw = fetch_csv(REDASH_DOMESTIC_URL, "Domestic Raw")
    df_port_trip = fetch_csv(REDASH_PORT_TRIP_URL, "Port Trip")
    df_port_direct = fetch_csv(REDASH_PORT_DIRECT_URL, "Port Direct")

    # Combine port data
    df_port_trip["source_file"] = "Port_Trip"
    df_port_direct["source_file"] = "Port_Direct"
    df_port_raw = pd.concat([df_port_trip, df_port_direct], ignore_index=True)
    print(f"\n📊 Combined port: {len(df_port_raw):,} records")

    # ── 2. Load static files ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📂 LOADING STATIC FILES")
    print("=" * 70)

    with open(DISTANCE_MATRIX_FILE, "rb") as f:
        distance_matrix = pickle.load(f)
    print(f"   Distance matrix: {len(distance_matrix):,} city pairs")

    city_df = pd.read_csv(CITY_REGIONS_FILE)
    variant_to_canonical = dict(zip(city_df["variant"], city_df["canonical"]))
    city_to_province = dict(zip(city_df["canonical"], city_df["province"]))
    city_to_region = dict(zip(city_df["canonical"], city_df["region"]))
    all_canonicals = set(city_df["canonical"].unique())
    print(f"   City normalization: {len(city_df)} variants → {len(all_canonicals)} canonicals")

    try:
        backhaul_df = pd.read_csv(BACKHAUL_FILE)
        print(f"   Backhaul: {len(backhaul_df)} cities")
    except FileNotFoundError:
        backhaul_df = pd.DataFrame()
        print("   Backhaul: Not found (optional)")

    # ── 3. Prepare domestic data ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🏗️  PREPARING DOMESTIC DATA")
    print("=" * 70)

    df_raw = clean_numeric_columns(
        df_raw, ["distance", "total_carrier_price", "total_shipper_price", "weight", "cost"]
    )
    if "cost" in df_raw.columns and "total_carrier_price" not in df_raw.columns:
        df_raw["total_carrier_price"] = df_raw["cost"]

    df_raw["pickup_date"] = pd.to_datetime(df_raw["pickup_date"], format="mixed", errors="coerce")
    print(f"   Raw: {len(df_raw):,} rows")
    print(f"   Date range: {df_raw['pickup_date'].min()} → {df_raw['pickup_date'].max()}")

    df = df_raw.copy()
    if "entity_mapping" in df.columns:
        df = df[df["entity_mapping"] == ENTITY_MAPPING]
    if "vehicle_type" in df.columns:
        df = df[df["vehicle_type"].isin(VEHICLE_TYPES)]
    df = df[
        (df["total_carrier_price"] >= MIN_CARRIER_PRICE)
        & (df["total_carrier_price"] <= MAX_CARRIER_PRICE)
    ]
    df = df[df["distance"] >= MIN_DISTANCE]
    df["cost_per_km"] = df["total_carrier_price"] / df["distance"]
    df = df[(df["cost_per_km"] >= MIN_CPK) & (df["cost_per_km"] <= MAX_CPK)]

    max_date = df["pickup_date"].max()
    df["days_ago"] = (max_date - df["pickup_date"]).dt.days
    df["lane"] = df["pickup_city"] + " → " + df["destination_city"]
    df["pickup_province"] = df["pickup_city"].map(city_to_province)
    df["dest_province"] = df["destination_city"].map(city_to_province)

    print(f"\n✅ Final domestic: {len(df):,} records | {df['lane'].nunique()} lanes")
    print(f"   Date range: {df['pickup_date'].min().date()} → {max_date.date()}")

    # ── 4. Build Index + Shrinkage model ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("📈 BUILDING INDEX + SHRINKAGE MODEL (90d HL)")
    print("=" * 70)

    decay_rate = np.log(2) / HALF_LIFE_DAYS
    df["weight"] = np.exp(-decay_rate * df["days_ago"])

    lane_counts = df.groupby("lane").size().sort_values(ascending=False)
    benchmark_lanes = list(lane_counts.head(N_BENCHMARK_LANES).index)

    benchmark_data = df[df["lane"].isin(benchmark_lanes)].copy()
    daily_index = benchmark_data.groupby(benchmark_data["pickup_date"].dt.date).apply(
        lambda g: np.average(g["cost_per_km"], weights=g["weight"]) if len(g) > 0 else np.nan
    ).dropna()
    smoothed_index = daily_index.rolling(INDEX_SMOOTHING_DAYS, min_periods=1).mean()
    current_index = smoothed_index.iloc[-1]
    print(f"   Current market index: {current_index:.4f} SAR/km")

    def calc_weighted_multiplier(group):
        if group["weight"].sum() == 0:
            return np.nan
        return np.average(group["cost_per_km"], weights=group["weight"]) / current_index if current_index > 0 else np.nan

    lane_multipliers = df.groupby("lane").apply(calc_weighted_multiplier).dropna().to_dict()

    def calc_weighted_stats(group):
        w_sum = group["weight"].sum()
        if w_sum == 0:
            return pd.Series({"lane_mean": np.nan, "lane_n": 0, "raw_n": len(group)})
        w_mean = np.average(group["cost_per_km"], weights=group["weight"])
        return pd.Series({"lane_mean": w_mean, "lane_n": w_sum, "raw_n": len(group)})

    lane_stats = df.groupby("lane").apply(calc_weighted_stats).to_dict("index")

    pickup_priors = df.groupby("pickup_city").apply(
        lambda g: np.average(g["cost_per_km"], weights=g["weight"]) if g["weight"].sum() > 0 else np.nan
    ).dropna().to_dict()

    dest_priors = df.groupby("destination_city").apply(
        lambda g: np.average(g["cost_per_km"], weights=g["weight"]) if g["weight"].sum() > 0 else np.nan
    ).dropna().to_dict()

    global_mean = np.average(df["cost_per_km"], weights=df["weight"])
    print(f"   Lane multipliers: {len(lane_multipliers)}")
    print(f"   Global weighted mean: {global_mean:.4f} SAR/km")

    # ── 5. Build Spatial model ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🗺️  BUILDING SPATIAL MODEL (R150 P50 IDW)")
    print("=" * 70)

    city_neighbors = defaultdict(list)
    for (c1, c2), dist in distance_matrix.items():
        if dist <= SPATIAL_RADIUS_KM * 2:
            city_neighbors[c1].append((c2, dist))
            city_neighbors[c2].append((c1, dist))

    for city in city_neighbors:
        city_neighbors[city] = sorted(set(city_neighbors[city]), key=lambda x: x[1])

    recent_df = df[df["days_ago"] <= SPATIAL_RECENCY_DAYS]

    city_outbound_cpk = recent_df.groupby("pickup_city").agg(
        outbound_cpk_mean=("cost_per_km", "mean"),
        outbound_count=("cost_per_km", "count"),
    ).to_dict("index")

    city_inbound_cpk = recent_df.groupby("destination_city").agg(
        inbound_cpk_mean=("cost_per_km", "mean"),
        inbound_count=("cost_per_km", "count"),
    ).to_dict("index")

    province_pair_cpk = (
        df.groupby(["pickup_province", "dest_province"])["cost_per_km"]
        .agg(["mean", "median", lambda x: np.percentile(x, SPATIAL_PERCENTILE)])
        .reset_index()
    )
    province_pair_cpk.columns = [
        "pickup_province", "dest_province", "mean", "median", f"p{SPATIAL_PERCENTILE}",
    ]
    province_pair_cpk = province_pair_cpk.set_index(
        ["pickup_province", "dest_province"]
    ).to_dict("index")

    print(f"   Cities with neighbors: {len(city_neighbors)}")
    print(f"   Province pairs: {len(province_pair_cpk)}")

    # ── 6. Package & export domestic models ───────────────────────────────────
    print("\n" + "=" * 70)
    print("💾 EXPORTING DOMESTIC MODELS")
    print("=" * 70)

    index_shrinkage_model = {
        "lane_multipliers": lane_multipliers,
        "lane_stats": lane_stats,
        "pickup_priors": pickup_priors,
        "dest_priors": dest_priors,
        "global_mean": global_mean,
        "current_index": current_index,
        "k": SHRINKAGE_K,
        "half_life_days": HALF_LIFE_DAYS,
        "training_date": str(max_date.date()),
        "n_lanes": len(lane_multipliers),
        "n_trips": len(df),
    }

    spatial_model = {
        "city_neighbors": dict(city_neighbors),
        "city_outbound_cpk": city_outbound_cpk,
        "city_inbound_cpk": city_inbound_cpk,
        "province_pair_cpk": province_pair_cpk,
        "city_to_province": city_to_province,
        "global_mean": global_mean,
        "config": {
            "radius_km": SPATIAL_RADIUS_KM,
            "percentile": SPATIAL_PERCENTILE,
            "min_neighbors": SPATIAL_MIN_NEIGHBORS,
        },
    }

    with open(os.path.join(EXPORT_DIR, "index_shrinkage_90hl.pkl"), "wb") as f:
        pickle.dump(index_shrinkage_model, f)
    with open(os.path.join(EXPORT_DIR, "spatial_r150_p50_idw.pkl"), "wb") as f:
        pickle.dump(spatial_model, f)
    with open(os.path.join(EXPORT_DIR, "distance_matrix.pkl"), "wb") as f:
        pickle.dump(distance_matrix, f)

    DISTANCE_LOOKUP = df[df["distance"] > 0].groupby("lane")["distance"].median().to_dict()
    BACKHAUL_LOOKUP = {}
    if "destination_city" in backhaul_df.columns and "backhaul_probability_pct" in backhaul_df.columns:
        BACKHAUL_LOOKUP = backhaul_df.set_index("destination_city")["backhaul_probability_pct"].to_dict()

    config = {
        "FEATURES": [
            "entity_mapping", "pickup_city", "destination_city",
            "commodity", "vehicle_type", "distance", "weight",
        ],
        "ENTITY_MAPPING": ENTITY_MAPPING,
        "DISTANCE_LOOKUP": DISTANCE_LOOKUP,
        "BACKHAUL_LOOKUP": BACKHAUL_LOOKUP,
        "RECENCY_CUTOFF_DAYS": RECENCY_CUTOFF_DAYS,
        "city_to_province": city_to_province,
        "city_to_region": city_to_region,
        "cascade": [
            {"priority": 1, "model": "Recency", "condition": f"Has loads in last {RECENCY_CUTOFF_DAYS} days"},
            {"priority": 2, "model": "Index+Shrinkage (90d HL)", "condition": "Has ANY historical data"},
            {"priority": 3, "model": "Spatial R150 P50 IDW", "condition": "New lane (no history)"},
        ],
    }
    with open(os.path.join(EXPORT_DIR, "config.pkl"), "wb") as f:
        pickle.dump(config, f)

    export_cols = [
        "pickup_date", "entity_mapping", "commodity", "vehicle_type",
        "pickup_city", "destination_city", "distance", "weight",
        "total_carrier_price", "days_ago", "lane", "cost_per_km",
        "pickup_province", "dest_province",
    ]
    for col in ["total_shipper_price", "container", "is_multistop"]:
        if col in df.columns:
            export_cols.append(col)
    df[[c for c in export_cols if c in df.columns]].to_csv(
        os.path.join(EXPORT_DIR, "reference_data.csv"), index=False
    )
    print("   ✅ Domestic models exported")

    # ── 7. Domestic holdout validation ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 DOMESTIC HOLDOUT VALIDATION")
    print("=" * 70)

    holdout = df[df["days_ago"] <= HOLDOUT_DAYS].copy()
    train = df[df["days_ago"] > HOLDOUT_DAYS].copy()

    holdout_lanes = holdout.groupby("lane").agg(
        actual_cpk=("cost_per_km", "mean"),
        trip_count=("cost_per_km", "count"),
        pickup_city=("pickup_city", "first"),
        dest_city=("destination_city", "first"),
    ).reset_index()

    lanes_with_history = set(train["lane"].unique())
    holdout_lanes["has_history"] = holdout_lanes["lane"].isin(lanes_with_history)

    print(f"   Holdout: {len(holdout):,} trips ({len(holdout_lanes)} lanes)")
    print(f"   WITH history: {holdout_lanes['has_history'].sum()}")
    print(f"   NEW lanes: {(~holdout_lanes['has_history']).sum()}")

    results = []
    for _, row in holdout_lanes.iterrows():
        pickup, dest = row["pickup_city"], row["dest_city"]
        actual = row["actual_cpk"]
        if row["has_history"]:
            pred, method = predict_index_shrinkage(pickup, dest, index_shrinkage_model)
        else:
            pred, method = predict_spatial(pickup, dest, spatial_model)
        results.append({
            "lane": row["lane"], "actual": actual, "predicted": pred,
            "method": method,
            "segment": "With History" if row["has_history"] else "New Lane",
            "error": pred - actual, "abs_error": abs(pred - actual),
        })

    results_df = pd.DataFrame(results)

    def calc_stats(subset, name):
        if len(subset) == 0:
            return None
        return {
            "Segment": name, "N": len(subset),
            "MAE": subset["abs_error"].mean(),
            "RMSE": np.sqrt((subset["error"] ** 2).mean()),
            "R²": r2_score(subset["actual"], subset["predicted"]) if len(subset) > 1 else np.nan,
            "±0.25": (subset["abs_error"] <= 0.25).mean() * 100,
            "±0.50": (subset["abs_error"] <= 0.50).mean() * 100,
        }

    stats_list = []
    for seg in ["With History", "New Lane"]:
        s = calc_stats(results_df[results_df["segment"] == seg], seg)
        if s:
            stats_list.append(s)
    stats_list.append(calc_stats(results_df, "OVERALL"))
    stats_df = pd.DataFrame(stats_list)
    print("\n" + stats_df.round(4).to_string(index=False))

    # ── 8. Port pricing ──────────────────────────────────────────────────────
    if len(df_port_raw) > 0:
        print("\n" + "=" * 70)
        print("🚢 PORT PRICING PIPELINE")
        print("=" * 70)

        # Validate cities
        validate_port_cities(df_port_raw, variant_to_canonical)

        # Clean
        df_port = df_port_raw.copy()
        for col in ["distance", "cost", "weight", "total_shipper_price", "total_carrier_price", "gb"]:
            if col in df_port.columns:
                df_port[col] = df_port[col].apply(
                    lambda x: pd.to_numeric(str(x).replace(",", ""), errors="coerce") if pd.notna(x) else np.nan
                )
        if "total_carrier_price" not in df_port.columns and "cost" in df_port.columns:
            df_port["total_carrier_price"] = df_port["cost"]

        df_port["pickup_date"] = pd.to_datetime(df_port["pickup_date"], format="mixed", errors="coerce")
        port_max_date = df_port["pickup_date"].max()
        df_port["days_ago"] = (port_max_date - df_port["pickup_date"]).dt.days

        if PORT_DAYS_LOOKBACK is not None:
            df_port = df_port[df_port["days_ago"] <= PORT_DAYS_LOOKBACK]

        # Load types
        df_port["is_roundtrip"] = df_port["pickup_city"] == df_port["destination_city"]
        fallback_type = np.where(df_port["is_roundtrip"], "Trip", "Direct")
        if "leg_name" in df_port.columns:
            df_port["load_type"] = np.where(df_port["leg_name"].isna(), fallback_type, df_port["leg_name"])
        else:
            df_port["load_type"] = fallback_type
        df_port["load_type"] = df_port["load_type"].replace({"Trip": "Trip", "Direct": "Direct"})
        df_port.loc[~df_port["load_type"].isin(["Trip", "Direct"]), "load_type"] = np.where(
            df_port.loc[~df_port["load_type"].isin(["Trip", "Direct"]), "is_roundtrip"], "Trip", "Direct"
        )

        # Effective destination
        def get_effective_dest(row):
            if row["load_type"] == "Trip" and pd.notna(row.get("true_drop_off")):
                tdo = str(row["true_drop_off"]).strip()
                if tdo:
                    return variant_to_canonical.get(tdo, tdo)
            return row["destination_city"]

        df_port["effective_dest"] = df_port.apply(get_effective_dest, axis=1)
        df_port["effective_lane"] = df_port["pickup_city"] + " → " + df_port["effective_dest"].astype(str)

        # Filters
        if "vehicle_type" in df_port.columns:
            df_port = df_port[df_port["vehicle_type"].isin(VEHICLE_TYPES)]
        df_port = df_port[
            (df_port["total_carrier_price"] >= PORT_MIN_CARRIER_PRICE)
            & (df_port["total_carrier_price"] <= PORT_MAX_CARRIER_PRICE)
        ]
        df_port = df_port[df_port["distance"] >= PORT_MIN_DISTANCE]
        df_port["cpk"] = df_port["total_carrier_price"] / df_port["distance"]
        df_port = df_port[(df_port["cpk"] >= PORT_MIN_CPK) & (df_port["cpk"] <= PORT_MAX_CPK)]
        if "id" in df_port.columns:
            df_port = df_port.drop_duplicates(subset=["id"], keep="first")

        print(f"   Final port dataset: {len(df_port):,} records")
        print(f"   Trip: {(df_port['load_type'] == 'Trip').sum():,} | Direct: {(df_port['load_type'] == 'Direct').sum():,}")

        # Domestic matching
        domestic_lane_ref = {}
        for lane, group in df.groupby("lane"):
            recent = group[group["days_ago"] <= 90]
            if len(recent) >= 3:
                domestic_lane_ref[lane] = {"median_cpk": recent["cost_per_km"].median(), "source": "recency"}
            else:
                domestic_lane_ref[lane] = {"median_cpk": group["cost_per_km"].median(), "source": "all_time"}
        global_cpk = df["cost_per_km"].median()

        def get_domestic_prediction(pickup, dest, distance_km):
            lane = f"{pickup} → {dest}"
            if lane in domestic_lane_ref:
                ref = domestic_lane_ref[lane]
                return ref["median_cpk"] * distance_km, f"Domestic_{ref['source']}"
            reverse = f"{dest} → {pickup}"
            if reverse in domestic_lane_ref:
                ref = domestic_lane_ref[reverse]
                return ref["median_cpk"] * distance_km, f"Domestic_reverse_{ref['source']}"
            return global_cpk * distance_km, "Global_CPK"

        port_results = [
            get_domestic_prediction(r["pickup_city"], r["effective_dest"], r["distance"])
            for _, r in df_port.iterrows()
        ]
        df_port["domestic_price"] = [r[0] for r in port_results]
        df_port["domestic_method"] = [r[1] for r in port_results]

        # Core lanes
        lane_volumes = (
            df_port.groupby("effective_lane")
            .agg(
                trips=("total_carrier_price", "count"),
                median_price=("total_carrier_price", "median"),
                dominant_type=("load_type", lambda x: x.value_counts().index[0]),
            )
            .sort_values("trips", ascending=False)
            .reset_index()
        )
        core_lanes = set(lane_volumes.head(PORT_CORE_TOP_N)["effective_lane"])
        df_port["is_core"] = df_port["effective_lane"].isin(core_lanes)
        print(f"   Core lanes: {len(core_lanes)} ({df_port['is_core'].mean()*100:.1f}% of volume)")

        # Fit linear transforms
        model_df = df_port[
            (df_port["domestic_price"].notna())
            & (df_port["domestic_price"] > 0)
            & (df_port["total_carrier_price"] > 0)
        ].copy()

        segments = [
            ("Overall", model_df),
            ("Core", model_df[model_df["is_core"]]),
            ("Non-Core", model_df[~model_df["is_core"]]),
            ("Trip", model_df[model_df["load_type"] == "Trip"]),
            ("Direct", model_df[model_df["load_type"] == "Direct"]),
            ("Core_Trip", model_df[(model_df["is_core"]) & (model_df["load_type"] == "Trip")]),
            ("Core_Direct", model_df[(model_df["is_core"]) & (model_df["load_type"] == "Direct")]),
            ("NonCore_Trip", model_df[(~model_df["is_core"]) & (model_df["load_type"] == "Trip")]),
            ("NonCore_Direct", model_df[(~model_df["is_core"]) & (model_df["load_type"] == "Direct")]),
        ]

        port_transforms = {}
        fit_results = []
        for seg_name, seg_df in segments:
            if len(seg_df) < 10:
                continue
            fit = fit_linear_transform(seg_df["domestic_price"], seg_df["total_carrier_price"])
            fit["segment"] = seg_name
            fit_results.append(fit)
            port_transforms[seg_name] = fit
            print(f"   {seg_name} (n={len(seg_df):,}): α={fit['alpha']:.4f}, β={fit['beta']:.0f}, MAE={fit['mae']:.0f}, R²={fit['r2']:.3f}")

        port_fit_results = pd.DataFrame(fit_results)

        # Export port model
        port_distance_lookup = df_port.groupby("effective_lane")["distance"].median().to_dict()
        port_distance_by_type = {
            "Trip": df_port[df_port["load_type"] == "Trip"].groupby("effective_lane")["distance"].median().to_dict(),
            "Direct": df_port[df_port["load_type"] == "Direct"].groupby("effective_lane")["distance"].median().to_dict(),
        }

        port_model = {
            "transforms": port_transforms,
            "recommended": {
                "Trip": port_transforms.get("Trip", port_transforms["Overall"]),
                "Direct": port_transforms.get("Direct", port_transforms["Overall"]),
                "fallback": port_transforms["Overall"],
            },
            "core_lanes": list(core_lanes),
            "lane_stats": df_port.groupby("effective_lane").agg(
                median_price=("total_carrier_price", "median"),
                median_cpk=("cpk", "median"),
                count=("total_carrier_price", "count"),
                dominant_type=("load_type", lambda x: x.value_counts().index[0]),
                avg_distance=("distance", "mean"),
            ).to_dict(orient="index"),
            "global_stats": {
                "Trip": {
                    "median_cpk": df_port[df_port["load_type"] == "Trip"]["cpk"].median(),
                    "count": int((df_port["load_type"] == "Trip").sum()),
                },
                "Direct": {
                    "median_cpk": df_port[df_port["load_type"] == "Direct"]["cpk"].median(),
                    "count": int((df_port["load_type"] == "Direct").sum()),
                },
            },
            "domestic_global_cpk": float(global_cpk),
            "training_date": str(port_max_date.date()),
            "n_records": len(df_port),
        }

        with open(os.path.join(EXPORT_DIR, "port_pricing_model.pkl"), "wb") as f:
            pickle.dump(port_model, f)

        port_distance_export = {"all_lanes": port_distance_lookup, "by_type": port_distance_by_type}
        with open(os.path.join(EXPORT_DIR, "port_distance_lookup.pkl"), "wb") as f:
            pickle.dump(port_distance_export, f)

        # Port reference data
        if "pickup_province" not in df_port.columns:
            df_port["pickup_province"] = df_port["pickup_city"].map(city_to_province)
        if "dest_province" not in df_port.columns:
            df_port["dest_province"] = df_port["destination_city"].map(city_to_province)
        if "lane" not in df_port.columns:
            df_port["lane"] = df_port["pickup_city"] + " → " + df_port["destination_city"]
        if "cost_per_km" not in df_port.columns:
            df_port["cost_per_km"] = df_port["cpk"]

        port_export_cols = [
            "pickup_date", "entity_mapping", "commodity", "vehicle_type",
            "pickup_city", "destination_city", "effective_dest", "effective_lane",
            "load_type", "distance", "weight", "total_carrier_price",
            "cpk", "cost_per_km", "days_ago", "lane",
            "pickup_province", "dest_province", "is_core",
            "domestic_price", "domestic_method",
        ]
        for col in ["total_shipper_price", "container", "is_multistop", "true_drop_off", "gb"]:
            if col in df_port.columns:
                port_export_cols.append(col)
        df_port[[c for c in port_export_cols if c in df_port.columns]].to_csv(
            os.path.join(EXPORT_DIR, "port_reference_data.csv"), index=False
        )
        port_fit_results.to_csv(os.path.join(EXPORT_DIR, "port_fit_results.csv"), index=False)
        print("   ✅ Port models exported")

    # ── 9. Write training metadata ────────────────────────────────────────────
    import json
    from datetime import datetime, timezone

    metadata = {
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "domestic_records": len(df),
        "domestic_lanes": df["lane"].nunique(),
        "domestic_date_range": f"{df['pickup_date'].min().date()} → {max_date.date()}",
        "overall_mae": round(float(stats_df[stats_df["Segment"] == "OVERALL"].iloc[0]["MAE"]), 4),
        "overall_r2": round(float(stats_df[stats_df["Segment"] == "OVERALL"].iloc[0]["R²"]), 4),
    }
    if len(df_port_raw) > 0 and "port_transforms" in dir():
        metadata["port_records"] = len(df_port)
        metadata["port_training_date"] = str(port_max_date.date())

    with open(os.path.join(EXPORT_DIR, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print(f"✅ TRAINING COMPLETE in {elapsed:.1f}s")
    print("=" * 70)
    print(f"\n📦 All artifacts saved to: {EXPORT_DIR}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
