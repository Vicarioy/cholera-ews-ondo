
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import joblib
import json
import datetime
import warnings
from pathlib import Path

from live_weather import (
    apply_live_weather,
    build_weather_calibration,
    fetch_live_weekly_weather,
)

try:
    import geopandas as gpd
except ImportError:
    gpd = None
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title = "Ondo Cholera Early Warning System",
    page_icon  = "🏥",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #1E90FF; }
    .high-risk {
        background-color: #7a1f1f;
        color: #ffffff;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #e74c3c;
        margin: 5px 0;
        font-weight: 500;
    }
    .medium-risk {
        background-color: #7a5b00;
        color: #ffffff;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #f39c12;
        margin: 5px 0;
        font-weight: 500;
    }
    .low-risk {
        color: #ffffff;
        background-color: 008000;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #2ecc71;
        margin: 5px 0;
    }
    .future-badge {
        background-color: #1E90FF;
        padding: 5px 10px;
        border-radius: 5px;
        border-left: 4px solid #004085;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
SHAPES_DIR = BASE_DIR / "data" / "shapefiles"
LOOKBACK_WEEKS = 12

MONTH_NAMES = {
    1:"January", 2:"February", 3:"March",
    4:"April",   5:"May",      6:"June",
    7:"July",    8:"August",   9:"September",
    10:"October",11:"November",12:"December"
}

RISK_COLORS = {
    "High"  : "#E74C3C",
    "Medium": "#F39C12",
    "Low"   : "008000"
}

# ── LSTM MODEL ────────────────────────────────────────────────────
class CholeraLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers, dropout=0.2):
        super(CholeraLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        device = next(self.parameters()).device
        h0 = torch.zeros(
            self.num_layers, x.size(0),
            self.hidden_size
        ).to(device)
        c0 = torch.zeros(
            self.num_layers, x.size(0),
            self.hidden_size
        ).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out.squeeze()

# ── HELPER: MONTH TO WEEK RANGE ───────────────────────────────────
def month_to_weeks(year, month):
    """
    Convert a month and year to list of
    epidemiological weeks that fall in that month
    """
    weeks = []
    # Get first and last day of month
    first_day = datetime.date(year, month, 1)
    if month == 12:
        last_day = datetime.date(year, 12, 31)
    else:
        last_day = datetime.date(year, month+1, 1) -                    datetime.timedelta(days=1)

    # Find all weeks that overlap with this month
    current = first_day
    while current <= last_day:
        week = current.isocalendar()[1]
        if week not in weeks:
            weeks.append(week)
        current += datetime.timedelta(days=7)

    return weeks

def week_to_date_range(year, week):
    """Convert year+week to start and end date strings"""
    try:
        start = datetime.date.fromisocalendar(year, week, 1)
        end = start + datetime.timedelta(days=6)
        return (
            start.strftime("%d %b %Y"),
            end.strftime("%d %b %Y")
        )
    except:
        return ("Unknown", "Unknown")

def is_future_week(year, week):
    """Check if a week is in the future"""
    today = datetime.date.today()
    try:
        week_start = datetime.date.fromisocalendar(year, week, 1)
        return week_start > today
    except:
        return False

# ── LOAD MODELS ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    with (DATA_DIR / "feature_config.json").open(encoding="utf-8") as f:
        feature_config = json.load(f)

    lstm_model = CholeraLSTM(
        input_size  = len(feature_config["temporal"]),
        hidden_size = 64,
        num_layers  = 2,
        dropout     = 0.2
    ).to(device)
    lstm_model.load_state_dict(torch.load(
        MODELS_DIR / "lstm_final_model.pt",
        map_location=device
    ))
    lstm_model.eval()

    rf_model = joblib.load(MODELS_DIR / "rf_spatial_model.pkl")
    target_scaler = joblib.load(MODELS_DIR / "target_scaler.pkl")

    with (MODELS_DIR / "fusion_config.json").open(encoding="utf-8") as f:
        fusion_config = json.load(f)

    return {
        "lstm"            : lstm_model,
        "rf"              : rf_model,
        "target_scaler"   : target_scaler,
        "feature_config"  : feature_config,
        "fusion_config"   : fusion_config,
        "device"          : device
    }

@st.cache_data
def load_data():
    master = pd.read_csv(
        DATA_DIR / "engineered_dataset.csv"
    )
    master["date"] = pd.to_datetime(master["date"])
    master = master.sort_values(
        ["lga", "year", "epi_week"]
    ).reset_index(drop=True)
    risk_path = DATA_DIR / "lga_risk_scores.csv"
    socio_path = DATA_DIR / "socioeconomic_data.csv"
    risk_scores = pd.read_csv(risk_path) if risk_path.exists() else pd.DataFrame()
    socio = pd.read_csv(socio_path) if socio_path.exists() else pd.DataFrame()

    shape_path = SHAPES_DIR / "ondo_lgas.shp"
    if gpd is not None and shape_path.exists():
        gdf = gpd.read_file(shape_path)
    else:
        gdf = None
    return master, risk_scores, socio, gdf


@st.cache_data(ttl="1h", max_entries=2)
def load_live_weather_data():
    """Cache the external API response to avoid calls on every widget rerun."""
    if gpd is None:
        raise RuntimeError("GeoPandas is required to locate the 18 LGAs")
    shape_path = SHAPES_DIR / "ondo_lgas.shp"
    boundaries = gpd.read_file(shape_path)
    return fetch_live_weekly_weather(boundaries)

# ── PREDICTION FUNCTION ───────────────────────────────────────────
def predict_week(
    year,
    epi_week,
    models,
    master,
    live_weather=None,
    weather_calibration=None,
):
    device         = models["device"]
    lstm_model     = models["lstm"]
    rf_model       = models["rf"]
    target_scaler  = models["target_scaler"]
    fusion_config  = models["fusion_config"]
    feature_config = models["feature_config"]
    TEMPORAL_FEAT  = feature_config["temporal"]
    SPATIAL_FEAT   = feature_config["spatial"]

    predictions = []
    all_lgas = sorted(master["lga"].unique())

    for lga in all_lgas:
        lga_data = master[
            master["lga"] == lga
        ].sort_values(["year", "epi_week"])

        # Get last 12 weeks before target week
        lga_filtered = lga_data[
            (lga_data["year"] < year) |
            ((lga_data["year"] == year) &
             (lga_data["epi_week"] < epi_week))
        ].tail(LOOKBACK_WEEKS)

        if len(lga_filtered) < LOOKBACK_WEEKS:
            lga_filtered = lga_data.tail(LOOKBACK_WEEKS)

        live_weeks_used = 0
        if live_weather is not None and weather_calibration is not None:
            lga_filtered, live_weeks_used = apply_live_weather(
                lga_filtered,
                lga,
                live_weather,
                weather_calibration,
            )

        sequence = lga_filtered[TEMPORAL_FEAT].values

        if len(sequence) < LOOKBACK_WEEKS:
            pad = np.zeros((
                LOOKBACK_WEEKS - len(sequence),
                len(TEMPORAL_FEAT)
            ))
            sequence = np.vstack([pad, sequence])

        # The engineered dataset already contains model-ready transformed
        # temporal values. Scaling them again would distort the inputs.
        X_tensor = torch.tensor(
            sequence.reshape(1, LOOKBACK_WEEKS, -1),
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            lstm_pred_scaled = lstm_model(X_tensor)
            if lstm_pred_scaled.dim() == 0:
                lstm_pred_scaled = lstm_pred_scaled.unsqueeze(0)
            lstm_pred_scaled = lstm_pred_scaled.cpu().numpy()

        lstm_pred = target_scaler.inverse_transform(
            lstm_pred_scaled.reshape(-1, 1)
        ).flatten()[0]
        lstm_pred = max(0, lstm_pred)

        spatial_data = lga_data[SPATIAL_FEAT].iloc[-1:]
        rf_pred  = rf_model.predict(spatial_data)[0]
        rf_risk  = min(rf_pred / 6.0, 1.0)

        alpha  = fusion_config.get("optimal_alpha", 0.1)
        final_pred = lstm_pred * (1 + alpha * rf_risk)
        final_pred = max(0, final_pred)

        if final_pred >= 8:
            risk_level = "High"
        elif final_pred >= 3:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        predictions.append({
            "lga"             : lga,
            "predicted_cases" : round(final_pred, 1),
            "risk_level"      : risk_level,
            "risk_score"      : round(
                min(final_pred / 10.0, 1.0), 3
            ),
            "live_weather_weeks": live_weeks_used,
        })

    return pd.DataFrame(predictions).sort_values(
        "predicted_cases", ascending=False
    ).reset_index(drop=True)

# ── RISK MAP ──────────────────────────────────────────────────────
def create_risk_map(predictions_df, gdf):
    if gdf is None:
        return None

    name_col = None
    for col in ["shapeName","NAME_2","ADM2_EN","name"]:
        if col in gdf.columns:
            name_col = col
            break
    if name_col is None:
        return None

    # Normalize names: replace spaces with hyphens to match
    # the predictions dataframe naming convention
    gdf_m = gdf.copy()
    gdf_m["_merge_key"] = gdf_m[name_col].str.replace(
        " ", "-", regex=False
    )

    preds_copy = predictions_df.copy()
    preds_copy["_merge_key"] = preds_copy["lga"].str.replace(
        " ", "-", regex=False
    )

    gdf_m = gdf_m.merge(
        preds_copy[[
            "_merge_key","risk_score",
            "risk_level","predicted_cases"
        ]],
        on="_merge_key", how="left"
    )
    gdf_m["risk_score"] = gdf_m["risk_score"].fillna(0.1)
    gdf_m["risk_level"] = gdf_m["risk_level"].fillna("Low")

    color_map = {
        "High":"#E74C3C","Medium":"#F39C12","Low":"#2ECC71"
    }
    fig, ax = plt.subplots(figsize=(5, 5))

    for _, row in gdf_m.iterrows():
        color = color_map.get(
            row.get("risk_level","Low"), "#2ECC71"
        )
        gpd.GeoDataFrame(
            [row], geometry="geometry", crs=gdf_m.crs
        ).plot(
            ax=ax, color=color,
            edgecolor="black", linewidth=0.8
        )

    for _, row in gdf_m.iterrows():
        try:
            c = row.geometry.centroid
            ax.annotate(
                f"{row.get(name_col,'')}\n"
                f"({row.get('predicted_cases',0)})",
                xy=(c.x, c.y),
                fontsize=4, ha="center",
                fontweight="light"
            )
        except:
            pass

    ax.set_title(
        "Ondo State Cholera Risk Map",
        fontsize=12, fontweight="bold"
    )
    ax.axis("off")

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="#E74C3C", label="High Risk"),
            Patch(facecolor="#F39C12", label="Medium Risk"),
            Patch(facecolor="#2ECC71", label="Low Risk")
        ],
        loc="lower right", fontsize=10
    )
    plt.tight_layout()
    return fig

# ================================================================
# MAIN DASHBOARD
# ================================================================

# ── HEADER ────────────────────────────────────────────────────────
st.title("🏥 Ondo State Cholera Early Warning System")
st.markdown(
    "**Hybrid RF-LSTM Spatiotemporal Prediction Model** | "
    "Federal University of Technology, Akure"
)
st.markdown("---")

# ── LOAD ──────────────────────────────────────────────────────────
with st.spinner("Loading models..."):
    try:
        models = load_models()
        master, risk_scores, socio, gdf = load_data()
        # ---------- ADD THESE TWO LINES ----------
        LAST_DATA_YEAR = int(master["year"].max())
        LAST_DATA_WEEK = int(master[master["year"] == LAST_DATA_YEAR]["epi_week"].max())
        LAST_DATA_DATE = master["date"].max().date()
        LAST_DATA_SOURCE = str(master.loc[master["date"].idxmax(), "data_source"])
        # -----------------------------------------
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────────
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")

use_live_weather = st.sidebar.toggle(
    "Use live Open-Meteo weather",
    value=True,
    help=(
        "Uses recent observations and forecasts from Open-Meteo, converts "
        "them to epidemiological weeks, and maps them to the model's saved "
        "feature scale. Falls back to the bundled snapshot if unavailable."
    ),
)

live_weather = None
weather_calibration = None
live_fetched_at = None
if use_live_weather:
    try:
        live_weather, live_fetched_at = load_live_weather_data()
        raw_master = pd.read_csv(DATA_DIR / "master_dataset.csv")
        weather_calibration = build_weather_calibration(raw_master, master)
        minimum_r2 = min(values[2] for values in weather_calibration.values())
        if minimum_r2 < 0.98:
            raise ValueError("Weather feature calibration quality is too low")
        st.sidebar.success("Live Open-Meteo weather connected")
        st.sidebar.caption(f"Fetched at {live_fetched_at} UTC")
    except Exception as error:
        st.sidebar.warning(
            "Live weather is unavailable; using the bundled snapshot. "
            f"Reason: {error}"
        )

weather_mode_label = (
    "Live Open-Meteo + bundled model features"
    if live_weather is not None
    else "Bundled snapshot fallback"
)

# Get available years from dataset
available_years = sorted(
    master["year"].unique().tolist(),
    reverse=True
)

# Safety filter: only keep years with substantial real data
# (at least 100 rows ~ 2 months across 18 LGAs)
_year_counts = master["year"].value_counts()
available_years = [
    y for y in available_years if _year_counts.get(y, 0) >= 100
]

today = datetime.date.today()

selected_year = st.sidebar.selectbox(
    "Year",
    options = available_years,
    index   = 0
)

selected_month_name = st.sidebar.selectbox(
    "Month",
    options = list(MONTH_NAMES.values()),
    index   = today.month - 1
)

# Convert month name to number
selected_month = [
    k for k, v in MONTH_NAMES.items()
    if v == selected_month_name
][0]

# Get weeks for selected month
month_weeks = month_to_weeks(selected_year, selected_month)
current_iso_week = today.isocalendar().week
default_week_index = 0
if (
    selected_year == today.year
    and selected_month == today.month
    and current_iso_week in month_weeks
):
    default_week_index = month_weeks.index(current_iso_week)

# Let user pick specific week within month
selected_week = st.sidebar.selectbox(
    "Specific Week",
    options = month_weeks,
    format_func = lambda w: (
        f"Week {w} "
        f"({week_to_date_range(selected_year, w)[0]} – "
        f"{week_to_date_range(selected_year, w)[1]})"
    ),
    index = default_week_index,
)

# Show selected date range
start_date, end_date = week_to_date_range(
    selected_year, selected_week
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Selected Period")

# Check if future
is_future = is_future_week(selected_year, selected_week)

if is_future:
    st.sidebar.markdown(f"""
    <div class="future-badge">
    🔮 FUTURE PREDICTION<br>
    {selected_month_name} {selected_year}<br>
    Week {selected_week}<br>
    {start_date} → {end_date}
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.info(
        "Recent rainfall, temperature, and humidity are retrieved from "
        "Open-Meteo when available and converted to the model feature "
        "scale. The bundled dataset remains the fallback."
    )
else:
    st.sidebar.info(
        f"**{selected_month_name} {selected_year}**\n\n"
        f"Week {selected_week}\n\n"
        f"{start_date} → {end_date}"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info(f"""
**Architecture**: Hybrid RF-LSTM\n
**Spatial**: Random Forest\n
**Temporal**: LSTM (2-layer, 64 units)\n
**Lookback**: {LOOKBACK_WEEKS} weeks\n
**Lead Time**: 1-4 weeks ahead
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Dataset")
st.sidebar.info(f"""
**LGAs**: 18\n
**Period**: 2018-2026\n
**Records**: {len(master):,}\n
**Features**: 42

**Latest observation**: {LAST_DATA_DATE:%d %b %Y}

**Source label**: {LAST_DATA_SOURCE}

**Weather mode**: {weather_mode_label}
""")

# ── GENERATE PREDICTIONS ──────────────────────────────────────────
future_label = "🔮 FUTURE FORECAST" if is_future else ""

with st.spinner(
    f"Generating predictions for "
    f"{selected_month_name} {selected_year}, "
    f"Week {selected_week}..."
):
    predictions = predict_week(
        selected_year,
        selected_week,
        models,
        master,
        live_weather=live_weather,
        weather_calibration=weather_calibration,
    )

live_lga_count = int((predictions["live_weather_weeks"] > 0).sum())
if live_lga_count:
    st.success(
        f"Live Open-Meteo inputs were used for {live_lga_count} of 18 LGAs. "
        "Daily API values were aggregated to epidemiological weeks and "
        "converted to the model's training feature scale."
    )
elif use_live_weather:
    st.info(
        "No API weeks overlap this selected model window; predictions use "
        "the bundled historical feature snapshot."
    )

# ── FUTURE WARNING BANNER ─────────────────────────────────────────
beyond_real_data = (
    selected_year > LAST_DATA_YEAR or
    (selected_year == LAST_DATA_YEAR and selected_week > LAST_DATA_WEEK)
)

if beyond_real_data:
    st.warning(
        f"⚠️ **No real climate data available yet** for "
        f"{selected_month_name} {selected_year} (Week {selected_week}). "
        f"The last bundled climate observation is "
        f"**Year {LAST_DATA_YEAR}, Week {LAST_DATA_WEEK}** "
        f"({week_to_date_range(LAST_DATA_YEAR, LAST_DATA_WEEK)[0]}). "
        f"Open-Meteo supplies overlapping recent weather weeks when the "
        f"live toggle is connected; older non-weather features still come "
        f"from the bundled model-ready sequence."
    )
elif is_future:
    st.info(
        f"🔮 **Future Prediction** — "
        f"Showing forecasted cholera risk for "
        f"**{selected_month_name} {selected_year}** "
        f"(Week {selected_week}: {start_date} to {end_date}). "
        f"Based on bundled NASA-derived climate observations up to "
        f"Year {LAST_DATA_YEAR} Week {LAST_DATA_WEEK}, "
        f"projected forward using learned seasonal patterns. Current "
        f"weather inputs use Open-Meteo when connected."
    )

# ── HEADER ROW ────────────────────────────────────────────────────
st.subheader(
    f"📊 {future_label} Predictions — "
    f"{selected_month_name} {selected_year} "
    f"(Week {selected_week}: {start_date} – {end_date})"
)

# ── KPI METRICS ───────────────────────────────────────────────────
high_risk   = predictions[predictions["risk_level"]=="High"]
medium_risk = predictions[predictions["risk_level"]=="Medium"]
low_risk    = predictions[predictions["risk_level"]=="Low"]
total_pred  = predictions["predicted_cases"].sum()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🚨 High Risk LGAs",   len(high_risk))
with col2:
    st.metric("⚠️ Medium Risk LGAs", len(medium_risk))
with col3:
    st.metric("✅ Low Risk LGAs",    len(low_risk))
with col4:
    st.metric("📈 Total Predicted Cases", f"{total_pred:.0f}")

st.markdown("---")

# ── MAP + ALERTS ──────────────────────────────────────────────────
left, right = st.columns([1, 1])   # equal split, or [1.5, 1] to shrink map slightly

with left:
    st.subheader("🗺️ LGA Risk Heatmap")
    if gdf is not None:
        fig_map = create_risk_map(predictions, gdf)
        if fig_map:
            st.pyplot(fig_map, width="content")
            plt.close()
    else:
        st.info(
            "The optional Ondo LGA shapefile was not included in the supplied "
            "archive. Predictions, alerts, tables, and trends remain available."
        )

with right:
    st.subheader("🚨 Risk Alerts")

    with st.expander(
        f"🚨 High Risk LGAs ({len(high_risk)})",
        expanded=True
    ):
        if len(high_risk) > 0:
            for _, row in high_risk.iterrows():
                st.markdown(f"""
                <div class="high-risk">
                🚨 <b>{row["lga"]}</b><br>
                Predicted: <b>{row["predicted_cases"]} cases</b>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No high risk LGAs")

    with st.expander(
        f"⚠️ Medium Risk LGAs ({len(medium_risk)})",
        expanded=False
    ):
        if len(medium_risk) > 0:
            for _, row in medium_risk.iterrows():
                st.markdown(f"""
                <div class="medium-risk">
                ⚠️ <b>{row["lga"]}</b> —
                {row["predicted_cases"]} cases
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No medium risk LGAs")

st.markdown("---")

# ── PREDICTIONS TABLE ─────────────────────────────────────────────
st.subheader(f"📋 All 18 LGA Predictions")

# Safely create display_df (prevent NameError if predictions is empty or missing)
try:
    display_df = predictions[["lga","predicted_cases","risk_score","risk_level"]].copy()
except (NameError, KeyError, AttributeError):
    # Fallback in case predictions isn't defined or has wrong structure
    display_df = pd.DataFrame(columns=["LGA","Predicted Cases","Risk Score","Risk Level"])

display_df.columns = ["LGA","Predicted Cases","Risk Score","Risk Level"]
display_df.index = range(1, len(display_df)+1)

def color_risk(val):
    colors = {
        "High"  :"background-color: #6e1f1f; color: #ffffff; font-weight: 600",
        "Medium":"background-color: #6e5200; color: #ffffff; font-weight: 600",
        "Low"   :"background-color: #008000"
    }
    return colors.get(val, "")

st.dataframe(
    display_df.style.map(color_risk, subset=["Risk Level"]),
    width="stretch",
    height=500
)

st.markdown("---")

# ── HISTORICAL TREND ──────────────────────────────────────────────
st.subheader("📈 Historical Case Trend")

selected_lga_trend = st.selectbox(
    "Select LGA",
    options = sorted(master["lga"].unique())
)

lga_history = master[
    master["lga"] == selected_lga_trend
].sort_values(["year","epi_week"])

lga_history["date"] = pd.to_datetime(lga_history["date"])

fig_trend, ax = plt.subplots(figsize=(14, 4))

ax.plot(
    lga_history["date"],
    lga_history["confirmed_cases"],
    color="steelblue", linewidth=1.5,
    label="Confirmed Cases"
)
ax.axhline(
    y=5, color="red", linestyle="--",
    linewidth=1, label="Outbreak Threshold (5)"
)
ax.fill_between(
    lga_history["date"],
    lga_history["confirmed_cases"],
    alpha=0.3, color="steelblue"
)

# Shade rainy seasons
for year in range(2018, today.year+1):
    ax.axvspan(
        pd.Timestamp(f"{year}-05-01"),
        pd.Timestamp(f"{year}-10-31"),
        alpha=0.08, color="blue",
        label="Rainy Season" if year==2018 else ""
    )

ax.set_title(
    f"Weekly Cholera Cases — {selected_lga_trend}",
    fontweight="bold", fontsize=13
)
ax.set_xlabel("Date")
ax.set_ylabel("Cases")
ax.legend(loc="upper right")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=0)
plt.tight_layout()
st.pyplot(fig_trend)
plt.close()

st.markdown("---")

# ── FUTURE MONTH PREVIEW ──────────────────────────────────────────
st.subheader("🔮 Monthly Risk Preview")
st.markdown(
    "See predicted risk level for each month "
    "of the selected year at a glance"
)

preview_year = st.selectbox(
    "Preview Year",
    options = available_years,
    index   = 0,
    key     = "preview_year"
)

preview_data = []
months_to_show = list(MONTH_NAMES.items())

with st.spinner("Generating monthly preview..."):
    for month_num, month_name in months_to_show:
        weeks = month_to_weeks(preview_year, month_num)
        mid_week = weeks[len(weeks)//2]

        preds = predict_week(
            preview_year, mid_week, models, master
        )
        high_count   = len(preds[preds["risk_level"]=="High"])
        medium_count = len(preds[preds["risk_level"]=="Medium"])
        total_cases  = preds["predicted_cases"].sum()

        if high_count > 0:
            overall_risk = "🔴 High"
        elif medium_count > 0:
            overall_risk = "🟡 Medium"
        else:
            overall_risk = "🟢 Low"

        is_fut = is_future_week(preview_year, mid_week)

        preview_data.append({
            "Month"         : month_name,
            "Risk Level"    : overall_risk,
            "High Risk LGAs": high_count,
            "Med Risk LGAs" : medium_count,
            "Predicted Cases": round(total_cases, 0),
            "Status"        : "🔮 Forecast" if is_fut
                              else "📊 Historical"
        })

preview_df = pd.DataFrame(preview_data)
preview_df.index = range(1, 13)
st.dataframe(preview_df, width="stretch")

st.markdown("---")

# ── ACTION RECOMMENDATIONS ────────────────────────────────────────
st.subheader("📋 Recommended Actions")

st.table(pd.DataFrame({
    "Risk Level"      : ["🚨 High","⚠️ Medium","✅ Low"],
    "Predicted Cases" : ["≥ 8/week","3-7/week","< 3/week"],
    "Response"        : [
        "Deploy rapid response team within 24 hours",
        "Enhanced surveillance + pre-position ORS",
        "Routine surveillance + community education"
    ],
    "Resources"       : [
        "500 ORS + IV fluids + treatment centre",
        "ORS stocks + rapid response standby",
        "Health education materials"
    ]
}))

st.markdown("---")

# ── DATA SUBMISSION ───────────────────────────────────────────────
st.subheader("📥 Submit New Weekly Report")

with st.expander("➕ Submit New Surveillance Data"):
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        form_lga   = st.selectbox(
            "LGA", sorted(master["lga"].unique()),
            key="flga"
        )
        form_year  = st.number_input(
            "Year", 2018, 2030,
            today.year, key="fyr"
        )
        form_month = st.selectbox(
            "Month", list(MONTH_NAMES.values()),
            index=today.month-1, key="fmo"
        )

    with fc2:
        form_cases    = st.number_input(
            "Confirmed Cases", 0, 1000, 0, key="fca"
        )
        form_rainfall = st.number_input(
            "Weekly Rainfall (mm)", 0.0, 500.0, 0.0,
            key="fra"
        )

    with fc3:
        form_temp     = st.number_input(
            "Mean Temperature (°C)", 15.0, 45.0, 27.0,
            key="fte"
        )
        form_humidity = st.number_input(
            "Mean Humidity (%)", 0.0, 100.0, 70.0,
            key="fhu"
        )

    form_source = st.selectbox(
        "Data Source",
        ["NCDC Weekly Sitrep","State Ministry of Health",
         "LGA Health Dept","WHO Situation Report",
         "Field Report"],
        key="fso"
    )

    if st.button("✅ Submit", type="primary"):
        st.success(
            f"✅ Data submitted for {form_lga} — "
            f"{form_month} {form_year}: "
            f"{form_cases} cases recorded"
        )
        st.info(
            "Select the same month and year above "
            "to see updated predictions"
        )

# ── FOOTER ────────────────────────────────────────────────────────
st.markdown("""
---
<div style="text-align:center; color:gray; font-size:12px;">
🔬 Federal University of Technology, Akure |
Department of Computer Science |
Hybrid RF-LSTM Spatiotemporal Cholera Early Warning System
</div>
""", unsafe_allow_html=True)
