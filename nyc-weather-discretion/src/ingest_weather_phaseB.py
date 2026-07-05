"""
ingest_weather_phaseB.py — Phase B: Open-Meteo Historical API から日次気象を取得する。

Phase A の src/ingest_weather.py と全く同じロジック（fetch_weather）を再利用し、
期間だけ config.yaml の phase_b.period に差し替える（地点・気象変数はトップレベル
weather 設定をそのまま流用、重複定義しない）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from src.ingest_weather import fetch_weather

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    df = fetch_weather(cfg["phase_b"]["period"], cfg["weather"])
    dest = PROJECT_ROOT / cfg["phase_b"]["paths"]["weather"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info("Saved %s  shape=%s", dest, df.shape)
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
