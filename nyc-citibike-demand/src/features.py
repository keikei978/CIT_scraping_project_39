"""
features.py — Phase 3: 特徴量生成。目的変数 y は需要そのもの（total/member/casual）に一本化。

列は接頭辞で群分けし、条件A/B/C/D は接頭辞選択で表現する（残差化は内部技法＝
暦特徴 cal_* がベースライン構造そのものなので、線形モデルに cal_* を与えること自体が
「残差予測＋復元」と代数的に等価。目的変数を需要 y のまま保てる＝リンゴ・ミカン比較を回避）。

  cal_*  : 暦特徴（trend, 曜日ダミー, 平滑季節 sin/cos, 祝日）。weatherを含まない。
  anom_* : 気象アノマリ（平年偏差, train のみで climatology を fit）。季節と直交。
  lag_*  : 目的変数 y の過去ラグ（deseason 後も残る短期自己相関を表す）。

条件:
  A = cal_*                （暦のみ＝ベースライン床）
  B = cal_* + anom_*       （+天候アノマリ）
  C = cal_* + lag_*        （+ラグ）
  D = cal_* + anom_* + lag_*（+両方）
"""

from __future__ import annotations

import logging
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import yaml

from src.baseline import weather_anomaly

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_features(merged: pd.DataFrame, target: str, cfg: dict) -> pd.DataFrame:
    df = merged.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"]

    train_end = pd.Timestamp(cfg["split"]["train_end"])
    train_mask = (dates <= train_end).to_numpy()

    out = pd.DataFrame({"date": dates, "y": df[target].to_numpy().astype(float)})

    # --- cal_*: 暦特徴（weatherを含まない・決定論的。モデルが係数を fit） ---
    day_index = (dates - dates.min()).dt.days.to_numpy().astype(float)
    out["cal_trend"] = day_index
    dow = dates.dt.dayofweek
    for d in range(1, 7):                      # 月曜=基準で drop_first
        out[f"cal_dow_{d}"] = (dow == d).astype(float)
    doy = dates.dt.dayofyear.to_numpy().astype(float)
    for k in range(1, 4):                       # 平滑季節（調和 3 次）
        out[f"cal_sin_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        out[f"cal_cos_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    yrs = dates.dt.year.unique().tolist()
    us = holidays.US(years=yrs)
    out["cal_is_holiday"] = dates.dt.date.map(lambda d: int(d in us)).astype(float)

    # --- anom_*: 気象アノマリ（train のみで climatology を fit, 季節と直交） ---
    nh = cfg["weather"]["anomaly"].get("harmonics", 3)
    for var in cfg["weather"]["daily_vars"]:
        out[f"anom_{var}"] = weather_anomaly(df, train_mask, var, n_harmonics=nh)

    # --- lag_*: 目的変数 y のラグ／rolling（正の shift のみ＝未来非参照） ---
    y = out["y"]
    for lag in cfg["features"]["lags"]:
        out[f"lag_y_{lag}"] = y.shift(lag)
    for w in cfg["features"]["rolling_windows"]:
        out[f"lag_roll_mean_{w}"] = y.shift(1).rolling(w).mean()

    return out


def condition_columns(cols: list[str], condition: str) -> list[str]:
    cal = [c for c in cols if c.startswith("cal_")]
    anom = [c for c in cols if c.startswith("anom_")]
    lag = [c for c in cols if c.startswith("lag_")]
    return {"A": cal, "B": cal + anom, "C": cal + lag, "D": cal + anom + lag}[condition]


def main() -> None:
    cfg = load_config()
    merged = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["merged"])
    fdir = PROJECT_ROOT / cfg["paths"]["features_dir"]
    fdir.mkdir(parents=True, exist_ok=True)

    targets = ["total", "member", "casual"] if cfg["citibike"]["keep_user_types"] else ["total"]
    for target in targets:
        feats = build_features(merged, target, cfg)
        dest = fdir / f"dataset_{target}.parquet"
        feats.to_parquet(dest, index=False)
        logger.info("Saved %s shape=%s cols=%d", dest.name, feats.shape, feats.shape[1])
    print("targets:", targets)
    print("example condition D cols:", condition_columns(list(feats.columns), "D"))


if __name__ == "__main__":
    main()
