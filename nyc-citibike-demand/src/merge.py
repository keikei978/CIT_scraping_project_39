"""
merge.py — Phase 2: 日次需要 × 日次気象を date で結合する。

  - 需要（daily_demand.parquet, [date, member, casual, total]）に
    気象（weather.parquet）を date で left join。
  - 需要・気象とも1日1行（システム全合算）なので 1:1 結合。
  - 出力: data/merged.parquet
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


def run_merge(cfg: dict) -> pd.DataFrame:
    demand = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["demand"])
    weather = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["weather"])
    demand["date"] = pd.to_datetime(demand["date"]).dt.normalize()
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()

    n = len(demand)
    merged = demand.merge(weather, on="date", how="left").sort_values("date")

    assert len(merged) == n, f"row count changed: {n} -> {len(merged)}"
    assert merged["date"].is_unique, "duplicate dates after merge"
    wcols = [c for c in weather.columns if c != "date"]
    miss = merged[wcols].isnull().sum()
    miss = miss[miss > 0]
    if not miss.empty:
        raise RuntimeError(f"weather missing after join:\n{miss.to_string()}")

    logger.info("merged: %d days, cols=%s", len(merged), list(merged.columns))
    return merged.reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    merged = run_merge(cfg)
    dest = PROJECT_ROOT / cfg["paths"]["merged"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(dest, index=False)
    logger.info("Saved %s  shape=%s", dest, merged.shape)
    print(merged.head().to_string(index=False))


if __name__ == "__main__":
    main()
