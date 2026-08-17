# Deploy to Streamlit Community Cloud

## 1. Upload to GitHub

Create a GitHub repository and place the **contents** of this folder at the repository root. The structure must include:

```text
requirements.txt
dashboard/app.py
data/processed/...
data/shapefiles/...
models/...
```

Do not upload `.venv`. Do not upload only `dashboard/app.py`; Streamlit Cloud needs the models, data, shapefile companions, and root `requirements.txt`.

## 2. Create the Streamlit app

1. Sign in at `share.streamlit.io`.
2. Select **Create app**.
3. Choose the GitHub repository and branch.
4. Set the main file path to `dashboard/app.py`.
5. Under advanced settings, select Python 3.13 if available.
6. Deploy and allow the first dependency installation to finish.

## 3. Verify the build

Open **Manage app → Logs**. Confirm that these packages install:

- PyTorch
- scikit-learn 1.6.1
- GeoPandas 1.1.1
- Streamlit 1.61.1

If `ModuleNotFoundError: geopandas` appears, confirm that `requirements.txt` is committed at the repository root and reboot the app.

## 4. Update behavior

The deployment rebuilds when the connected GitHub branch changes. It does not read files from Google Drive or Colab. Recent weather is requested from Open-Meteo at runtime and cached for one hour; no API secret is required. The committed datasets remain the fallback and provide non-weather model features.

## 5. Resource note

The PyTorch dependency and model loading can make the first cloud start slow. This is normal. If the free cloud instance exceeds its resource limit, use a CPU-only PyTorch installation strategy supported by the deployment environment or host the application on a larger service.
