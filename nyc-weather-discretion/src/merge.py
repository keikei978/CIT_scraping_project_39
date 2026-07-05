"""
merge.py — セグメント別日次件数（long）に日次気象を date で結合する。
出力: data/merged.parquet  [date, segment, n, <weather vars...>]
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_merge(paths: dict) -> pd.DataFrame:
    seg = pd.read_parquet(PROJECT_ROOT / paths["segments"])
    wx = pd.read_parquet(PROJECT_ROOT / paths["weather"])
    seg["date"] = pd.to_datetime(seg["date"]).dt.normalize()
    wx["date"] = pd.to_datetime(wx["date"]).dt.normalize()

    merged = seg.merge(wx, on="date", how="left")
    wcols = [c for c in wx.columns if c != "date"]
    miss = merged[wcols].isnull().sum()
    miss = miss[miss > 0]
    if not miss.empty:
        raise RuntimeError(f"weather missing after join:\n{miss.to_string()}")
    logger.info("merged: %d rows, segments=%d", len(merged), merged["segment"].nunique())
    return merged


def main() -> None:
    cfg = load_config()
    merged = run_merge(cfg["paths"])
    dest = PROJECT_ROOT / cfg["paths"]["merged"]
    merged.to_parquet(dest, index=False)
    logger.info("Saved %s shape=%s", dest, merged.shape)


if __name__ == "__main__":
    main()
