"""
train.py — Phase 4: 特徴アブレーション × 3モデルの学習・評価（目的変数は需要 y）。

  - 目的変数は需要 y（total/member/casual）。残差化は cal_* 特徴として内包（リンゴ・ミカン回避）。
  - 条件 A/B/C/D は features.condition_columns で接頭辞選択。
  - モデル: Linear(=Ridge, scaler内包) / RandomForest / XGBoost。CV は TimeSeriesSplit。
  - BASE 行: 暦ベースライン（baseline.fit_baseline）だけの床も記録。
  - 評価は需要スケールで MAE/RMSE/R²（val でハイパラ選択、test は最終報告のみ）。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform, randint, uniform
from xgboost import XGBRegressor

from src.baseline import fit_baseline
from src.features import condition_columns
from src.split import load_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _metrics(y, p, prefix):
    return {
        f"{prefix}_mae": mean_absolute_error(y, p),
        f"{prefix}_rmse": math.sqrt(mean_squared_error(y, p)),
        f"{prefix}_r2": r2_score(y, p),
    }


def _search(model, params, n_iter, cv, rs):
    return RandomizedSearchCV(model, params, n_iter=n_iter, cv=cv,
                              scoring="neg_mean_absolute_error",
                              random_state=rs, n_jobs=-1, refit=True)


def run_target(target: str, cfg: dict) -> list[dict]:
    train, val, test = load_splits(target, cfg)
    n_iter = cfg["tuning"]["n_iter"]
    rs = cfg["tuning"]["random_state"]
    cv = TimeSeriesSplit(n_splits=cfg["tuning"]["cv_splits"])
    rows = []

    # --- BASE: 暦ベースラインのみ（weatherもラグも使わない床） ---
    alldf = pd.concat([train, val, test]).sort_values("date").reset_index(drop=True)
    tmask = (alldf["date"] <= pd.Timestamp(cfg["split"]["train_end"])).to_numpy()
    base = fit_baseline(alldf.rename(columns={"y": "y"}), tmask, "y")
    alldf["_base"] = base
    vb = alldf[alldf["date"].isin(val["date"])]
    tb = alldf[alldf["date"].isin(test["date"])]
    rows.append({"target": target, "condition": "BASE", "model": "Baseline",
                 **_metrics(val["y"], vb["_base"], "val"),
                 **_metrics(test["y"], tb["_base"], "test")})

    all_cols = list(train.columns)
    for condition in cfg["conditions"]:
        feats = condition_columns(all_cols, condition)
        Xtr, ytr = train[feats], train["y"]
        Xva, Xte = val[feats], test[feats]

        models = {
            "Linear": _search(
                Pipeline([("sc", StandardScaler()), ("m", Ridge())]),
                {"m__alpha": loguniform(1e-2, 1e4)}, n_iter, cv, rs),
            "RandomForest": _search(
                RandomForestRegressor(random_state=rs, n_jobs=-1),
                {"n_estimators": randint(100, 500), "max_depth": randint(2, 12),
                 "min_samples_leaf": randint(1, 20)}, n_iter, cv, rs),
            "XGBoost": _search(
                XGBRegressor(objective="reg:squarederror", random_state=rs,
                             n_jobs=-1, verbosity=0),
                {"n_estimators": randint(50, 400), "max_depth": randint(2, 7),
                 "learning_rate": loguniform(0.01, 0.3),
                 "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4),
                 "reg_alpha": loguniform(1e-4, 10), "reg_lambda": loguniform(1e-4, 10)},
                n_iter, cv, rs),
        }
        for mname, search in models.items():
            search.fit(Xtr, ytr)
            rows.append({"target": target, "condition": condition, "model": mname,
                         **_metrics(val["y"], search.predict(Xva), "val"),
                         **_metrics(test["y"], search.predict(Xte), "test")})
            logger.info("[%s] %s/%s done", target, condition, mname)
    return rows


def main() -> None:
    cfg = load_config()
    targets = ["total", "member", "casual"] if cfg["citibike"]["keep_user_types"] else ["total"]
    all_rows = []
    for t in targets:
        all_rows.extend(run_target(t, cfg))

    df = pd.DataFrame(all_rows)
    num = [c for c in df.columns if c.endswith(("_mae", "_rmse", "_r2"))]
    df[num] = df[num].round(2)
    dest = PROJECT_ROOT / cfg["paths"]["results_dir"] / "ablation_table.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    logger.info("Saved %s", dest)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
