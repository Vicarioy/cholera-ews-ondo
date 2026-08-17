# Cholera Early Warning System — Ondo State, Nigeria

**Spatiotemporal ML-based cholera outbreak prediction for 18 LGAs in Ondo State.**

Final Year Project — Dept. of Computer Science, FUTA, 2025

## Authors
- Oluwagbenga Victor Daniel (CSC/20/4873)
- Popoola Moses Eniola (CSC/20/4881)

## Supervisor
Dr. I. T. Jimoh

## Model Performance
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest (spatial) | 0.1281 | 0.2050 | 0.9785 |
| LSTM (temporal) | 0.5635 | 0.9251 | 0.7719 |
| Hybrid RF-LSTM (final) | 0.5724 | — | 0.7833 |

**Outbreak detection (threshold = 5 cases/week):**
Accuracy 96% · Precision 89% · Recall 55%

## Spatial Analysis
- Global Moran's I = 0.2351 (p = 0.0360) — significant clustering
- LISA Hotspots: Okitipupa, Irele
- Cold spots: Akoko cluster (north)

## Structure
## Run the Dashboard
```bash
pip install streamlit
streamlit run dashboard/app.py
```

## Stack
Python · PyTorch · scikit-learn · SHAP · GeoPandas · Streamlit · NASA POWER API
