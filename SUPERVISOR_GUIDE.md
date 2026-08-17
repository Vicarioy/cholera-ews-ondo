# Independent supervisor review guide

You can inspect and run this project independently, without the student being present.

## A. Fastest review on Windows

1. Clone the GitHub repository, or extract the ZIP completely; do not run files from inside the ZIP preview.
2. Open the project folder containing `requirements.txt`.
3. Install 64-bit Python 3.13 if needed and select **Add Python to PATH** during installation.
4. Double-click `setup_windows.bat` and wait for `SETUP COMPLETE`.
5. Double-click `run_dashboard.bat` and leave its console window open.
6. The launcher opens `http://localhost:8501`; enter that address manually if the browser does not open.
7. Stop the application with `Ctrl+C` in the launcher window.

No Google Drive, Colab mount, API key, or database is required.

## B. Code inspection checklist

Open `dashboard/app.py` in VS Code and inspect these sections:

1. **Imports and optional GeoPandas import** — verifies graceful behavior if mapping support is unavailable.
2. **Constants/path block** — uses `Path(__file__).resolve().parents[1]`, so no Colab `/content/drive` paths remain.
3. **`CholeraLSTM`** — two LSTM layers, 64 hidden units, dropout, and a linear output.
4. **`load_models()`** — loads the LSTM state dictionary, Random Forest, target scaler, feature order, and fusion configuration.
5. **`load_data()`** — loads the bundled engineered dataset and shapefile.
6. **`predict_week()`** — creates the 12-week sequence, performs LSTM and RF inference, applies multiplicative fusion, and classifies risk.
7. **`create_risk_map()`** — joins LGA predictions to the shapefile and renders the risk map.
8. **Main dashboard** — filters, metrics, alerts, tables, historical trends, and preview.

## C. Files that should be verified

- `models/lstm_final_model.pt`
- `models/rf_spatial_model.pkl`
- `models/target_scaler.pkl`
- `models/fusion_config.json`
- `data/processed/feature_config.json`
- `data/processed/engineered_dataset.csv`
- All five `data/shapefiles/ondo_lgas.*` files
- `requirements.txt`

## D. Suggested technical checks

From PowerShell in the project folder:

```powershell
.\.venv\Scripts\python.exe -m py_compile dashboard\app.py
.\.venv\Scripts\python.exe verify_project.py
.\.venv\Scripts\python.exe -c "import streamlit, torch, sklearn, geopandas; print(streamlit.__version__, torch.__version__, sklearn.__version__, geopandas.__version__)"
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

The expected application signals are:

- “Models loaded successfully”
- 18 LGAs in the dataset
- Prediction metrics and the all-LGA table
- A rendered Ondo LGA risk map
- No `/content/drive` or `MyDrive` error

## E. What can and cannot be supervised

The following can be reviewed:

- Dashboard source code
- Model architecture used for inference
- Model-loading and fusion logic
- Processed feature names and data
- Saved evaluation metadata and figures
- Local and cloud execution

The following cannot be fully reproduced from this repository:

- Raw-data acquisition
- Original feature-engineering execution
- Model training and hyperparameter search
- Independent regeneration of the supplied model artifacts

Those require the original training notebook/scripts and raw-data sources, which were not included in either supplied ZIP.

## F. Launcher troubleshooting

If `run_dashboard.bat` closes or localhost does not open:

1. Confirm that the repository was fully cloned or extracted and that `run_dashboard.bat`, `setup_windows.bat`, and `requirements.txt` are in the same folder.
2. Run `setup_windows.bat` again and confirm that it ends with `SETUP COMPLETE`.
3. Open PowerShell in the project folder and run:

```powershell
.\run_dashboard.bat
```

4. Read `dashboard_startup.log` in the project folder. The improved launcher records missing-environment and Streamlit startup errors there and pauses instead of silently disappearing.
5. If port 8501 is already occupied, stop the other Streamlit window with `Ctrl+C`, then launch again.
6. Verify the installation directly:

```powershell
.\.venv\Scripts\python.exe verify_project.py
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.address localhost --server.port 8501
```

Do not run `run_dashboard.bat` directly from GitHub's web page or from inside a ZIP preview.

## G. Weather-data interpretation

`dashboard/live_weather.py` calls Open-Meteo without an API key. It retrieves recent/forecast daily rainfall, mean temperature, and mean humidity for representative points inside all 18 LGA polygons, aggregates the values into epidemiological weeks, and transforms them to the saved model scale. `dashboard/app.py` overlays matching API weeks on the 12-week LSTM sequence. If the API fails, the dashboard visibly falls back to the bundled feature snapshot.

The live API does not provide cholera case surveillance or socioeconomic attributes. Those inputs remain from the bundled validated dataset. This separation prevents raw weather values from being inserted directly into a model trained on transformed features.

## H. Interpreting an all-Low screen

Risk classification is performed separately for every LGA:

- Low: predicted cases below 3
- Medium: predicted cases from 3 up to (but not including) 8
- High: predicted cases of 8 or more

It is possible for the state total to exceed 18 or 30 cases while every individual LGA remains below 3. Review the numerical LGA table before interpreting the colored category. Do not lower thresholds solely to make the dashboard display warning colors; threshold changes require epidemiological justification and validation.
