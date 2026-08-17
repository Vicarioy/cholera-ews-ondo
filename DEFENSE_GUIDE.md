# Defense guide: Ondo State Cholera Early Warning System

## 1. One-minute project explanation

This project is a cholera early-warning dashboard for the 18 Local Government Areas of Ondo State. It combines two machine-learning models. An LSTM learns temporal patterns from the preceding 12 epidemiological weeks, including reported cases and environmental variables. A Random Forest models spatial vulnerability using relatively stable LGA characteristics such as population density, water access, sanitation, open defecation, poverty, elevation, and distance to water bodies. Their outputs are fused to estimate cases and assign Low, Medium, or High risk. Streamlit presents the predictions as indicators, alerts, tables, and historical trends.

## 2. The problem being solved

Cholera risk varies across both time and location. A purely temporal model may detect recent case and weather patterns but miss structural differences between LGAs. A purely spatial model may identify vulnerable locations but miss rapidly changing outbreak signals. The hybrid approach captures both dimensions.

## 3. Inputs and outputs

### Temporal branch — LSTM

- Input window: 12 previous epidemiological weeks.
- Input size: 24 features per week.
- Architecture recovered from the supplied state dictionary: two LSTM layers, 64 hidden units, dropout of 0.2, followed by one linear output.
- Important feature groups: rainfall, temperature, humidity, lagged cases, lagged weather, four-week rolling values, month/rainy-season indicators, and cyclical week encodings.
- Output: a scaled numerical case prediction, converted back to the original case scale by `target_scaler.pkl`.

### Spatial branch — Random Forest

The eight feature names are stored in the supplied Random Forest artifact:

1. Population density
2. Water access
3. Sanitation access
4. Open defecation
5. Multidimensional Poverty Index (MPI)
6. Elevation
7. Distance to coast
8. Distance to river

The Random Forest output is converted into a bounded spatial-risk factor.

### Fusion and classification

The supplied `fusion_config.json` specifies multiplicative fusion with `alpha = 0.1`:

```text
final prediction = LSTM prediction × (1 + alpha × RF risk)
```

Risk categories in the dashboard are:

- High: predicted cases ≥ 8
- Medium: predicted cases ≥ 3 and < 8
- Low: predicted cases < 3

The supplied fusion configuration reports MAE 0.571, RMSE 0.910, and R² 0.779. Present these as metadata saved with the supplied model, not as metrics independently reproduced by this local conversion.

## 4. What each metric means

- **MAE (Mean Absolute Error):** the average absolute difference between actual and predicted cases. Lower is better.
- **RMSE (Root Mean Squared Error):** similar to MAE but penalizes large errors more strongly. Lower is better.
- **R² (coefficient of determination):** the proportion of outcome variation explained by the model. Values nearer 1 indicate a better fit.

## 5. Why LSTM?

LSTM is designed for sequences. Its memory gates help retain useful information across multiple weeks and reduce the vanishing-gradient problem found in basic recurrent networks. A 12-week window gives the system recent epidemiological and environmental context for the target week.

## 6. Why Random Forest?

Random Forest combines many decision trees, handles nonlinear relationships and interactions, works well with tabular data, and is less sensitive to feature scaling. It is suitable for LGA-level socioeconomic and geographic characteristics.

## 7. Why hybrid fusion?

The temporal estimate is the primary prediction. The spatial model modifies that estimate based on local vulnerability. With multiplicative fusion, the same temporal signal can lead to a slightly higher estimate in a more vulnerable LGA.

## 8. Data pipeline

```text
Raw surveillance + weather + socioeconomic/geographic data
                         ↓
Cleaning and LGA/week alignment
                         ↓
Lag, rolling, seasonal, and cyclical feature engineering
                         ↓
12-week temporal sequences ──→ LSTM ──┐
                                      ├─→ fused prediction → risk level
LGA spatial attributes ─────→ RF ─────┘
                                      ↓
                             Streamlit dashboard
```

The local archive begins at the processed-data and trained-model stages; it does not contain the original raw-data/training notebook.

## 9. What was changed for local execution

### Before

The app used a Colab-specific path:

```python
BASE_DIR = "/content/drive/MyDrive/Cholera_EWS/"
```

That path exists only inside Google Colab after Google Drive is mounted.

### After

The app calculates its project folder from its own file location:

```python
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
```

This makes the project portable: it can be moved to another Windows folder without editing paths.

Other local changes include CPU-safe PyTorch loading, version-pinned dependencies, portable shapefile handling, and Windows setup/run scripts.

## 10. Compatibility reconstruction that must be disclosed

The ZIP omitted several artifacts expected by `app.py`. The conversion made the following transparent, evidence-based reconstruction:

- The LSTM state dictionary shows an input width of 24. The 24 temporal variables were inferred from the engineered dataset and saved in `feature_config.json` in that exact order.
- The Random Forest exposes its eight exact feature names through `feature_names_in_`.
- Inspection showed that `engineered_dataset.csv` already contains transformed/model-ready temporal and spatial columns. The original dashboard loaded temporal and spatial scaler files but did not use them during prediction. The consolidated app therefore does not apply a second transformation, which would distort the model inputs.
- The missing scaler files are not required by the repaired inference path. `target_scaler.pkl` is retained because it is used to convert the LSTM output back to the case-count scale.
- Missing risk-score and socioeconomic CSVs were not used elsewhere in the original dashboard, so they are optional.
- The complete five-file Ondo LGA shapefile bundle was recovered from `cholera_ews_complete.zip`, merged, and tested with GeoPandas.

Do not claim that this reconstruction exactly reproduces the original training pipeline. Exact reproduction requires the original preprocessing files and notebook.

## 11. Demonstration procedure

1. Run `run_dashboard.bat`.
2. Explain that Streamlit starts a local server at `http://localhost:8501`.
3. Show the latest data year and epidemiological week in the sidebar.
4. Select a year, month, and week.
5. Explain the four indicators: number of high-, medium-, and low-risk LGAs and total predicted cases.
6. Open the High and Medium risk alert panels.
7. Show the table for all 18 LGAs.
8. Select an LGA in the historical trend section and explain the actual time series.
9. Show the monthly preview and recommended public-health actions.
10. Show the LGA map and explain that its geometry comes from the bundled `.shp`, `.shx`, `.dbf`, `.prj`, and `.cpg` files.

## 12. Likely defense questions and concise answers

### Is this a diagnostic system?

No. It is a decision-support and early-warning tool. It estimates population-level risk and does not diagnose individual patients.

### Why use the previous 12 weeks?

It provides roughly three months of recent epidemiological and environmental context while keeping the sequence manageable. The supplied trained LSTM architecture expects 12-week sequences in this dashboard.

### How do you prevent negative case predictions?

After inverse scaling, the dashboard clips predictions at zero because negative disease counts are not meaningful.

### Does the dashboard retrain the model?

No. It loads already-trained artifacts and performs inference. Retraining requires the original training notebook and raw/preprocessing pipeline.

### Why use robust scaling?

RobustScaler uses the median and interquartile range, making it less sensitive to outbreak spikes and other extreme observations than mean/standard-deviation scaling.

### What is data leakage?

Data leakage occurs when information unavailable at prediction time influences training or preprocessing. A fully reproducible study should fit every preprocessing object only on the original training split and should preserve those fitted objects with the model.

### What are the main limitations?

- Predictions depend on the quality and timeliness of surveillance data.
- Some data are literature-derived or extended rather than live reports.
- The supplied package omitted the original training notebook and some preprocessing artifacts.
- The current dashboard uses fixed risk thresholds.
- The original training notebook and original feature configuration were not supplied, so full training reproduction is not yet possible.
- Model performance can change over time and should be monitored and recalibrated.

### How would you improve the system?

- Add automated, validated NCDC and weather feeds.
- Recover and version the complete preprocessing/training pipeline.
- Use strict time-based train/validation/test splits and rolling-origin evaluation.
- Add uncertainty intervals and calibration analysis.
- Validate the bundled LGA geometry and name matching against an authoritative current boundary source.
- Monitor drift and retrain periodically with newly confirmed data.
- Add authentication and an audit trail before operational deployment.

## 13. Safe claims for your presentation

You can say:

- “I converted the supplied inference dashboard from Colab-specific paths to portable local paths.”
- “The system combines temporal LSTM predictions with Random Forest spatial vulnerability.”
- “The app runs locally and does not require Google Drive.”
- “I identified and documented missing preprocessing artifacts instead of hiding them.”

Avoid saying:

- “I reproduced the complete training pipeline,” because the training notebook was not supplied.
- “The supplied archive reproduces the complete training process.” The training notebook was not included.
- “The tool guarantees an outbreak.” It provides estimated risk for decision support.
