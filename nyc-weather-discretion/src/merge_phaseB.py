"""
merge_phaseB.py — Phase B: 地下鉄セグメント別日次件数（long）に日次気象を date で結合する。
出力: data/phaseB_merged.parquet  [date, segment, n, <weather vars...>]

src/merge.py の run_merge() をそのまま再利用し、paths だけ phase_b の値に差し替える。
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from src.merge import run_merge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    pb_paths = cfg["phase_b"]["paths"]
    merged = run_merge({
        "segments": pb_paths["subway_segments"],
        "weather": pb_paths["weather"],
        "merged": pb_paths["merged"],
    })
    dest = PROJECT_ROOT / pb_paths["merged"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(dest, index=False)
    logger.info("Saved %s shape=%s", dest, merged.shape)
    print(merged.groupby("segment").agg(days=("n", "size"), mean_per_day=("n", "mean")).round(0).to_string())


if __name__ == "__main__":
    main()
