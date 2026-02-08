# Freight Pricing — Daily Retrain Pipeline

Automated daily retraining of domestic and port freight pricing models for Saudi Arabia logistics.

## Architecture

```
├── train.py                                        # Production training script
├── requirements.txt
├── .github/workflows/retrain.yml                   # Daily cron at 08:00 AST
└── pricing_app_final/model_export/                  # Static data + auto-updated models
    ├── distance_matrix.pkl                          # (static — manual updates)
    ├── city_normalization_with_regions.csv           # (static — manual updates)
    ├── Backhaul Prob.csv                            # (static — manual updates)
    ├── index_shrinkage_90hl.pkl                     # (auto-updated)
    ├── spatial_r150_p50_idw.pkl
    ├── config.pkl
    ├── reference_data.csv
    ├── port_pricing_model.pkl
    ├── port_distance_lookup.pkl
    ├── port_reference_data.csv
    ├── port_fit_results.csv
    └── training_metadata.json
```

## Data Sources (Redash)

| Dataset | Query |
|---------|-------|
| Domestic Raw | `4869` |
| Port Roundtrip | `4871` |
| Port Direct | `4870` |

## Setup

1. Ensure your static files are in `pricing_app_final/model_export/`:
   - `distance_matrix.pkl`
   - `city_normalization_with_regions.csv`
   - `Backhaul Prob.csv`

2. Push to GitHub — the workflow runs automatically at **08:00 AST** daily, or trigger manually via **Actions → Daily Model Retrain → Run workflow**.

3. The Streamlit app reads models directly from `pricing_app_final/model_export/`.

## Cascade Order

1. **Recency** → loads in last 90 days
2. **Index + Shrinkage (90d HL)** → any historical data
3. **Spatial R150 P50 IDW** → new lanes
4. **Port Transform** → α × Domestic + β (port loads)
