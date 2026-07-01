"""
split.py — Phase 4: 日付による厳密な時系列分割（ランダム分割禁止）。

  train: date <= split.train_end
  val:   train_end < date <= split.val_end
  test:  date > val_end
ラグ由来の NaN は dropna（全条件で同じ行が落ちる＝対称）。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_splits(target: str = "total", cfg: dict | None = None):
    if cfg is None:
        cfg = load_config()
    path = PROJECT_ROOT / cfg["paths"]["features_dir"] / f"dataset_{target}.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().reset_index(drop=True)

    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])
    train = df[df["date"] <= train_end].copy()
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test = df[df["date"] > val_end].copy()

    assert len(train) and len(val) and len(test), "empty split"
    assert train["date"].max() < val["date"].min(), "train/val overlap"
    assert val["date"].max() < test["date"].min(), "val/test overlap"
    return train, val, test


if __name__ == "__main__":
    for t in ("total", "member", "casual"):
        try:
            tr, va, te = load_splits(t)
            print(f"[{t}] train={len(tr)} val={len(va)} test={len(te)} "
                  f"| {tr['date'].min().date()}..{te['date'].max().date()}")
        except FileNotFoundError:
            print(f"[{t}] dataset not found yet")
