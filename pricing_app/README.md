# 🚚 Freight Pricing Negotiation Tool

A web-based tool for sales teams to get instant pricing guidance for freight negotiations.

## Features

- **Dropdown Selection**: Pick cities, vehicle types, and commodities from lists
- **Smart Defaults**: Commodity and weight auto-detect if not specified
- **Negotiation Corridor**: Anchor (start), Target (fair), Floor (walk-away) prices
- **Recent Market Prices**: Min/Max from last 90 days with exact load details
- **Ammunition**: Historical similar loads to cite in negotiations (time-spread)
- **No Code Required**: Clean interface for sales team
- **Fast Loading**: Uses pre-trained models (no training on cloud)

## Project Structure

```
pricing_app/
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── model_export/          # Pre-trained models (from notebook)
    ├── carrier_model.cbm  # CatBoost carrier price model
    ├── shipper_model.cbm  # CatBoost shipper price model
    ├── knn_bundle.pkl     # KNN model + encoders + scaler
    ├── reference_data.parquet  # Historical loads for KNN
    └── config.pkl         # Configuration and distance lookup
```

## Setup Workflow

### Step 1: Train Models Locally (One-Time)

1. Open `pricing_negotiation_pipeline_v2.ipynb` in Jupyter
2. Run all cells including the **"Export Models for Deployment"** section at the end
3. This creates the `model_export/` folder with all trained artifacts

### Step 2: Prepare for Deployment

Copy the exported models to the app folder:

```bash
cp -r model_export/ pricing_app/
```

Your folder should now look like:
```
pricing_app/
├── app.py
├── requirements.txt
├── model_export/
│   ├── carrier_model.cbm
│   ├── shipper_model.cbm
│   ├── knn_bundle.pkl
│   ├── reference_data.parquet
│   └── config.pkl
```

### Step 3: Deploy to Streamlit Cloud

1. Create a GitHub repository
2. Push the `pricing_app/` folder contents (including `model_export/`):

```bash
cd pricing_app
git init
git add .
git commit -m "Initial commit with pre-trained models"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/pricing-tool.git
git push -u origin main
```

3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Sign in with GitHub
5. Click "New app" → Select your repo → Deploy

Your app will be live at `https://YOUR_APP_NAME.streamlit.app`

## Running Locally

```bash
cd pricing_app
pip install -r requirements.txt
streamlit run app.py
```

## Updating the Models

When you have new data and want to retrain:

1. Update your CSV data file
2. Re-run the notebook (all cells)
3. The export section will regenerate `model_export/`
4. Copy the new `model_export/` to your git repo
5. Push to GitHub → Streamlit Cloud auto-redeploys

## Model Files Explained

| File | Size | Description |
|------|------|-------------|
| `carrier_model.cbm` | ~200KB | CatBoost model predicting carrier costs |
| `shipper_model.cbm` | ~200KB | CatBoost model predicting shipper prices |
| `knn_bundle.pkl` | ~50KB | KNN model + label encoders + scaler |
| `reference_data.parquet` | ~100KB | Historical loads for finding similar |
| `config.pkl` | ~10KB | Feature lists, distance lookup, settings |

## Security Notes

- The app is read-only (no data modification)
- Model files contain aggregated statistics, not raw business data
- For internal use, add authentication via Streamlit Cloud settings
- Or deploy behind your company VPN

## Troubleshooting

**"Missing model files" error:**
- Make sure you ran the notebook's export section
- Check that `model_export/` folder is in the same directory as `app.py`
- All 5 files must be present

**Slow first load:**
- First load caches the models (5-10 seconds)
- Subsequent loads are instant

**Predictions seem off:**
- Check how old your training data is (shown in "Model Information")
- Retrain with fresh data if needed

## Support

For issues or feature requests, contact [your team].
