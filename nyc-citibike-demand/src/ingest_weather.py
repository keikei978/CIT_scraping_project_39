"""
ingest_weather.py — Phase 2: Open-Meteo Historical API から日次気象を取得する。

  - NYC 重心の単一座標（全合算需要のため1点でよい）。
  - config.weather.daily_vars を取得し data/weather.parquet に保存。
  - 生値を保存し、アノマリ化（平年偏差）は features.py で train のみで推定する
    （climatology を train のみで fit＝リーク防止のため、取得段階では生値のみ）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_weather(cfg: dict) -> pd.DataFrame:
    w = cfg["weather"]
    params = {
        "latitude": w["latitude"],
        "longitude": w["longitude"],
        "start_date": cfg["period"]["start_date"],
        "end_date": cfg["period"]["end_date"],
        "daily": ",".join(w["daily_vars"]),
        "timezone": "America/New_York",
    }
    logger.info("Fetching Open-Meteo daily weather %s ~ %s",
                params["start_date"], params["end_date"])
    r = requests.get(w["archive_url"], params=params, timeout=180)
    r.raise_for_status()
    daily = r.json()["daily"]

    df = pd.DataFrame(daily).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # 欠損があれば報告（archive-api は通常全日揃う）。サイレントに埋めない。
    miss = df[w["daily_vars"]].isnull().sum()
    miss = miss[miss > 0]
    if not miss.empty:
        raise RuntimeError(f"weather has missing values:\n{miss.to_string()}")

    logger.info("weather: %d days, vars=%s", len(df), w["daily_vars"])
    return df[["date"] + w["daily_vars"]]


def main() -> None:
    cfg = load_config()
    df = fetch_weather(cfg)
    dest = PROJECT_ROOT / cfg["paths"]["weather"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info("Saved %s  shape=%s", dest, df.shape)
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
