# Ondo State Cholera Early Warning System

This is the consolidated review and deployment package. It combines the repaired local/Streamlit Cloud dashboard, trained RF–LSTM artifacts, processed datasets, the complete Ondo LGA shapefile bundle, and the original analytical figures.

Final Year Project — Department of Computer Science, Federal University of Technology, Akure.

**Authors:** Oluwagbenga Victor Daniel (CSC/20/4873) and Popoola Moses Eniola (CSC/20/4881)  
**Supervisor:** Dr. S.A Adeleye

## Saved model performance metadata

The supplied fusion configuration reports MAE 0.571, RMSE 0.910, and R² 0.779. These are saved training/evaluation metadata, not metrics recalculated by the deployed dashboard.

## What the application does

The dashboard loads a two-layer LSTM for 12-week temporal patterns and a Random Forest for LGA-level spatial vulnerability. It combines their outputs using the fusion configuration in `models/fusion_config.json`, then displays predicted cases, risk categories, alerts, trends, monthly previews, and an LGA map.

## Live weather and model integration

The dashboard calls the no-key Open-Meteo Forecast API for all 18 LGA representative points. Daily rainfall, mean temperature, and mean humidity are aggregated into ISO epidemiological weeks. The application derives the original raw-to-model affine transformations from the overlapping `master_dataset.csv` and `engineered_dataset.csv` records; all 14 weather transformations reproduce the saved model features with R² = 1.0. Matching API weeks are overlaid on the 12-week LSTM input sequence.

The API response is cached for one hour. If Open-Meteo is unavailable or the selected historical window does not overlap the API period, the application falls back to the bundled model-ready sequence and says so visibly. Surveillance/case and spatial vulnerability features remain from the validated bundled dataset; the API supplies weather only.

## Why the current screen may show all LGAs as Low

The default model output can legitimately place all 18 LGAs below the fixed Medium threshold of 3 predicted cases per LGA, even when the total across the state is larger. In the verified August 2026 example, the model loaded successfully and produced a state total while each individual LGA remained below 3. The dashboard must not relabel predictions merely to create Medium or High alerts.

The current-week selector defaults to the actual ISO week. A low result is not equivalent to “weather alone says risk is low”; it means the saved model, available live-weather overlay or fallback sequence, fusion rule, and fixed thresholds produced a low category.

## Run on Windows

1. Install 64-bit Python 3.13 and select **Add Python to PATH**.
2. Double-click `setup_windows.bat` once.
3. Double-click `run_dashboard.bat` whenever you want to run the application.
4. Open `http://localhost:8501` if the browser does not open automatically.
5. Keep the terminal window open. Press `Ctrl+C` to stop the server.

Equivalent PowerShell commands:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

## Main files

```text
dashboard/app.py                 Streamlit interface and inference logic
data/processed/                  Model-ready datasets and feature order
data/shapefiles/                 Complete Ondo LGA shapefile bundle
models/                          Trained models, target scaler, fusion config
outputs/figures/                 Original analytical and evaluation figures
requirements.txt                 Reproducible Python dependencies
SUPERVISOR_GUIDE.md              Independent inspection and review procedure
DEPLOY_STREAMLIT_CLOUD.md        GitHub/Streamlit Cloud deployment procedure
DEFENSE_GUIDE.md                 Presentation and likely defense questions
verify_project.py                Offline integrity and artifact checks
```

## Reproducibility boundary

The package supports inspection and execution of the dashboard/inference code. It does not include the original model-training notebook or raw-data ingestion pipeline. Therefore, a reviewer can reproduce the supplied predictions but cannot independently retrain the models from raw data using this archive alone.

After setup, run the independent integrity checks with:

```powershell
.\.venv\Scripts\python.exe verify_project.py
```
