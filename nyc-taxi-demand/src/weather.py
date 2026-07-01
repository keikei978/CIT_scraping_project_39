"""
weather.py - Open-Meteo archive-api から日次気象データを取得する

エンドポイント: https://archive-api.open-meteo.com/v1/archive
period.start_date ~ end_date の全日分を取得し、
date をキーにした DataFrame を返す。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Open-Meteo archive-api から日次気象データを取得して DataFrame を返す。

    Returns
    -------
    DataFrame with:
      - date (datetime64[ns])
      - temperature_2m_max, temperature_2m_min, precipitation_sum,
        snowfall_sum, windspeed_10m_max  (config.weather.daily_vars に従う)
    """
    period_cfg = cfg["period"]
    weather_cfg = cfg["weather"]

    start_date = period_cfg["start_date"]
    end_date = period_cfg["end_date"]
    latitude = weather_cfg["latitude"]
    longitude = weather_cfg["longitude"]
    daily_vars = weather_cfg["daily_vars"]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(daily_vars),
        "timezone": "America/New_York",
    }

    logger.info(
        "Fetching weather from archive-api: %s ~ %s, vars=%s",
        start_date,
        end_date,
        daily_vars,
    )

    try:
        resp = requests.get(ARCHIVE_API_URL, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error("Weather API request failed: %s", e)
        raise

    daily_data = data.get("daily", {})
    if not daily_data or "time" not in daily_data:
        raise ValueError(f"Unexpected weather API response structure: {list(data.keys())}")

    weather_df = pd.DataFrame(daily_data)
    weather_df = weather_df.rename(columns={"time": "date"})
    weather_df["date"] = pd.to_datetime(weather_df["date"])

    # 期間内に絞る（API が返す範囲を念のため確認）
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    weather_df = weather_df[
        (weather_df["date"] >= start_ts) & (weather_df["date"] <= end_ts)
    ].reset_index(drop=True)

    logger.info(
        "Weather data fetched: shape=%s, date range=%s ~ %s",
        weather_df.shape,
        weather_df["date"].min(),
        weather_df["date"].max(),
    )

    # 欠損値の報告
    missing = weather_df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.warning("Weather data has missing values:\n%s", missing.to_string())
    else:
        logger.info("No missing values in weather data.")

    return weather_df


if __name__ == "__main__":
    cfg = load_config()
    df = fetch_weather(cfg)
    print("\n=== weather summary ===")
    print(f"shape   : {df.shape}")
    print(f"columns : {df.columns.tolist()}")
    print(df.head(5).to_string(index=False))
