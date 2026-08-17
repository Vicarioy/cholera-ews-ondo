"""Live Open-Meteo weather retrieval and model-feature adaptation."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


API_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_FEATURES = [
    "rainfall_weekly",
    "temp_mean",
    "humidity_mean",
    "rainfall_weekly_lag1",
    "rainfall_weekly_lag2",
    "rainfall_weekly_lag3",
    "rainfall_weekly_lag4",
    "temp_mean_lag1",
    "temp_mean_lag2",
    "temp_mean_lag4",
    "humidity_mean_lag1",
    "humidity_mean_lag2",
    "rainfall_weekly_roll4",
    "rainfall_weekly_roll4_sum",
]


def _normalise_lga(name: str) -> str:
    return str(name).strip().replace(" ", "-")


def _lga_key(name: str) -> str:
    """Match harmless space, hyphen, slash, and case differences."""
    return "".join(character for character in str(name).lower() if character.isalnum())


def fetch_live_weekly_weather(gdf, timeout: int = 30) -> tuple[pd.DataFrame, str]:
    """Fetch recent/forecast daily weather for all LGA representative points."""
    name_col = next(
        (column for column in ["shapeName", "NAME_2", "ADM2_EN", "name"] if column in gdf),
        None,
    )
    if name_col is None:
        raise ValueError("No supported LGA name column exists in the shapefile")

    points = gdf.geometry.representative_point()
    lgas = [_normalise_lga(name) for name in gdf[name_col]]
    params = {
        "latitude": ",".join(f"{point.y:.5f}" for point in points),
        "longitude": ",".join(f"{point.x:.5f}" for point in points),
        "daily": (
            "temperature_2m_mean,precipitation_sum,"
            "relative_humidity_2m_mean"
        ),
        "past_days": 92,
        "forecast_days": 16,
        "timezone": "Africa/Lagos",
    }
    response = requests.get(API_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    locations = payload if isinstance(payload, list) else [payload]
    if len(locations) != len(lgas):
        raise ValueError("Open-Meteo returned an unexpected number of locations")

    frames = []
    for lga, location in zip(lgas, locations):
        daily = pd.DataFrame(location["daily"])
        daily["date"] = pd.to_datetime(daily["time"])
        daily["lga"] = lga
        frames.append(daily)

    daily = pd.concat(frames, ignore_index=True)
    iso = daily["date"].dt.isocalendar()
    daily["year"] = iso.year.astype(int)
    daily["epi_week"] = iso.week.astype(int)
    weekly = (
        daily.groupby(["lga", "year", "epi_week"], as_index=False)
        .agg(
            rainfall_weekly=("precipitation_sum", "sum"),
            temp_mean=("temperature_2m_mean", "mean"),
            humidity_mean=("relative_humidity_2m_mean", "mean"),
            days=("date", "count"),
            week_start=("date", "min"),
            week_end=("date", "max"),
        )
        .sort_values(["lga", "year", "epi_week"])
        .reset_index(drop=True)
    )
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return weekly, fetched_at


def _raw_weather_features(master: pd.DataFrame) -> pd.DataFrame:
    frame = master.sort_values(["lga", "year", "epi_week"]).copy()
    grouped = frame.groupby("lga", group_keys=False)
    for lag in (1, 2, 3, 4):
        frame[f"rainfall_weekly_lag{lag}"] = grouped["rainfall_weekly"].shift(lag)
    for lag in (1, 2, 4):
        frame[f"temp_mean_lag{lag}"] = grouped["temp_mean"].shift(lag)
    for lag in (1, 2):
        frame[f"humidity_mean_lag{lag}"] = grouped["humidity_mean"].shift(lag)
    frame["rainfall_weekly_roll4"] = grouped["rainfall_weekly"].transform(
        lambda values: values.rolling(4, min_periods=1).mean()
    )
    frame["rainfall_weekly_roll4_sum"] = grouped["rainfall_weekly"].transform(
        lambda values: values.rolling(4, min_periods=1).sum()
    )
    return frame


def build_weather_calibration(
    master: pd.DataFrame, engineered: pd.DataFrame
) -> dict[str, tuple[float, float, float]]:
    """Infer affine raw-to-model transformations from overlapping saved data."""
    raw = _raw_weather_features(master)
    keys = ["year", "epi_week", "lga"]
    joined = raw[keys + WEATHER_FEATURES].merge(
        engineered[keys + WEATHER_FEATURES],
        on=keys,
        suffixes=("_raw", "_model"),
        how="inner",
    )
    calibration = {}
    for feature in WEATHER_FEATURES:
        pair = joined[[f"{feature}_raw", f"{feature}_model"]].dropna()
        x = pair[f"{feature}_raw"].to_numpy(dtype=float)
        y = pair[f"{feature}_model"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        ss_total = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - float(np.sum((y - predicted) ** 2)) / ss_total
        calibration[feature] = (float(slope), float(intercept), r_squared)
    return calibration


def apply_live_weather(
    sequence_rows: pd.DataFrame,
    lga: str,
    live_weekly: pd.DataFrame,
    calibration: dict[str, tuple[float, float, float]],
) -> tuple[pd.DataFrame, int]:
    """Overlay API-derived, training-scale weather features on a model sequence."""
    live = live_weekly[
        live_weekly["lga"].map(_lga_key) == _lga_key(lga)
    ].copy()
    if live.empty:
        return sequence_rows, 0

    live = live.rename(columns={"days": "weather_days"})
    live = _raw_weather_features(live)
    for feature in WEATHER_FEATURES:
        slope, intercept, _ = calibration[feature]
        live[feature] = live[feature] * slope + intercept

    overlay = live[["year", "epi_week", "weather_days"] + WEATHER_FEATURES]
    result = sequence_rows.merge(
        overlay,
        on=["year", "epi_week"],
        how="left",
        suffixes=("", "_live"),
    )
    used = result["weather_days"].notna()
    for feature in WEATHER_FEATURES:
        live_column = f"{feature}_live"
        result.loc[used & result[live_column].notna(), feature] = result.loc[
            used & result[live_column].notna(), live_column
        ]
    drop_columns = ["weather_days"] + [f"{feature}_live" for feature in WEATHER_FEATURES]
    return result.drop(columns=drop_columns), int(used.sum())
