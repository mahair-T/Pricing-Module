# 🚚 Freight Pricing Negotiation Tool

ML-powered pricing tool for freight logistics.

## Features

- **ML Model**: CatBoost regression for carrier and shipper price prediction
- **Simplified KNN**: Recent loads sorted by match quality (same lane, commodity, recency)
- **Blended Pricing**: Combines model predictions with historical actuals
- **Negotiation Corridor**: Anchor / Target / Ceiling prices

## Files

```
pricing_app/
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── model_export/
    ├── carrier_model.json    # Carrier price model (JSON format)
    ├── shipper_model.json    # Shipper price model (JSON format)
    ├── reference_data.csv    # Historical loads for KNN
    └── config.pkl            # Configuration and lookups
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run locally:
```bash
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Connect repo to [share.streamlit.io](https://share.streamlit.io)
3. Set main file path: `app.py`

## Configuration

Edit `config.pkl` to change:
- `FEATURES`: Model input features
- `RECENCY_CUTOFF_DAYS`: Days for "recent" data (default: 90)
- `ENTITY_MAPPING`: Entity type (default: Domestic)
- `DISTANCE_LOOKUP`: Lane distance mapping
- `BACKHAUL_LOOKUP`: City backhaul probabilities

## KNN Logic (Simplified)

1. **Filter**: Vehicle type (required) + last N days
2. **Score**: Same lane (+100) + Same commodity (+10) + Same container (+5) - days_ago*0.5
3. **Sort**: By score descending, then by recency
4. **Fallback**: If no recent matches, show older data

## Pricing Logic

- Recent data + diverges >30% → 80% actuals + 20% model
- Recent data + close to model → 50/50 blend
- Historical only → 60% historical + 40% model
- No history → Model only
