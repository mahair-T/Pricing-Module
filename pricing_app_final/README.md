# Freight Pricing Negotiation Tool

ML-powered carrier price predictions with negotiation corridors.

## Performance
- **Overall:** 88.1% within 20%
- **Ports:** 89.6% within 20%
- **Domestic:** 86.8% within 20%

## Features

### ML Features (in model)
- entity_mapping, commodity, vehicle_type, cities, distance, weight
- **container** (0/1) - containerized cargo
- **is_multistop** (0/1) - auto-detected from same-city + distance>100km

### Rare Lane Logic (not in model)
- Lanes with <10 historical loads use **backhaul-based pricing**
- Finds similar lanes by destination backhaul probability
- Computes GB/km from similar lanes → applies to rare lane

## Required Files

```
pricing_app/
├── app.py                    # Streamlit app
├── requirements.txt          # Pinned dependencies (catboost==1.2.8)
├── model_export/
│   ├── carrier_model.json    # CatBoost model (JSON format - portable)
│   ├── shipper_model.json    # CatBoost model (JSON format - portable)
│   ├── carrier_model.cbm     # CatBoost model (binary backup)
│   ├── shipper_model.cbm     # CatBoost model (binary backup)
│   ├── knn_bundle.pkl        # KNN + scaler + encoders
│   ├── config.pkl            # Config with lookups
│   └── reference_data.parquet # Historical data for matching
```

## Deployment

1. Create GitHub repo with all files above
2. Go to share.streamlit.io
3. Connect repo, set main file: `app.py`
4. Deploy

**Note:** App will try JSON models first (most portable), falls back to CBM if needed.

## Usage

**Required inputs:**
- Entity Mapping (Domestic/Ports/etc)
- Pickup City
- Destination City  
- Vehicle Type

**Optional (auto-detected):**
- Container (0/1)
- Commodity
- Weight

## Match Priority

1. **EXACT** ✅ - Same entity + lane + commodity + vehicle + container + multistop
2. **LANE** 🔶 - Same entity + lane + container + multistop (different commodity/vehicle)
3. **SIMILAR** 🔄 - Same entity + container, different lane (distance-based)

## Troubleshooting

If CatBoost errors occur:
- Ensure `catboost==1.2.8` in requirements.txt
- App loads JSON format which is more version-tolerant
- Check Streamlit Cloud logs for details
