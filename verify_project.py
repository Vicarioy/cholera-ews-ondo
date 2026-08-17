"""Offline integrity checks for the consolidated inference project."""

from pathlib import Path
import json

import geopandas as gpd
import joblib
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
required = [
    ROOT / "dashboard" / "app.py",
    ROOT / "data" / "processed" / "engineered_dataset.csv",
    ROOT / "data" / "processed" / "feature_config.json",
    ROOT / "data" / "shapefiles" / "ondo_lgas.shp",
    ROOT / "data" / "shapefiles" / "ondo_lgas.shx",
    ROOT / "data" / "shapefiles" / "ondo_lgas.dbf",
    ROOT / "data" / "shapefiles" / "ondo_lgas.prj",
    ROOT / "data" / "shapefiles" / "ondo_lgas.cpg",
    ROOT / "models" / "lstm_final_model.pt",
    ROOT / "models" / "rf_spatial_model.pkl",
    ROOT / "models" / "target_scaler.pkl",
    ROOT / "models" / "fusion_config.json",
]

missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
assert not missing, f"Missing files: {missing}"

app_source = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
assert "/content/drive" not in app_source and "MyDrive" not in app_source

config = json.loads(
    (ROOT / "data" / "processed" / "feature_config.json").read_text(encoding="utf-8")
)
assert len(config["temporal"]) == 24
assert len(config["spatial"]) == 8

data = pd.read_csv(ROOT / "data" / "processed" / "engineered_dataset.csv")
assert len(data) == 8010
assert data["lga"].nunique() == 18

boundaries = gpd.read_file(ROOT / "data" / "shapefiles" / "ondo_lgas.shp")
assert len(boundaries) == 18

rf_model = joblib.load(ROOT / "models" / "rf_spatial_model.pkl")
assert list(rf_model.feature_names_in_) == config["spatial"]

lstm_state = torch.load(
    ROOT / "models" / "lstm_final_model.pt",
    map_location="cpu",
    weights_only=True,
)
assert tuple(lstm_state["lstm.weight_ih_l0"].shape) == (256, 24)

print("PASS: all required files are present")
print("PASS: zero Colab/Google Drive paths")
print("PASS: 8,010 model-ready rows across 18 LGAs")
print("PASS: 18 LGA boundary geometries")
print("PASS: Random Forest feature order matches configuration")
print("PASS: LSTM state dictionary expects 24 temporal inputs")
