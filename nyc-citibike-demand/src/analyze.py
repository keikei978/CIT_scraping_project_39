"""
analyze.py — Phase 5: 本研究の主役図表を生成する（精度の最大化ではなく寄与の特定）。

出力:
  results/standardized_coef.csv  : 条件D・線形モデルの標準化係数（ラグ vs 天候アノマリ vs 暦）。
                                   ★同一目的変数・同一モデル内でのみ比較（精査 critique-6）。
  results/residual_anomaly.csv   : 残差 r=y−baseline と 天候アノマリ の相関（「顕在化」を正直に示す）。
  results/badweather_eval.csv    : 天候ショック日に限定した C(ラグのみ) vs D(ラグ+天候) のMAE比較
                                   （精査 enhance-2: lag-1 が外す「前日から急変した日」で weather の真価）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.baseline import fit_baseline
from src.features import condition_columns
from src.split import load_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALPHAS = np.logspace(-2, 4, 30)


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _group(col: str) -> str:
    return {"cal": "calendar", "anom": "weather_anom", "lag": "lag"}[col.split("_")[0]]


def standardized_coefs(target, cfg):
    train, _, _ = load_splits(target, cfg)
    feats = condition_columns(list(train.columns), "D")
    pipe = Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=ALPHAS))]).fit(
        train[feats], train["y"])
    coef = pipe.named_steps["m"].coef_
    return pd.DataFrame({"target": target, "feature": feats, "group": [_group(c) for c in feats],
                         "std_coef": coef}).assign(abs=lambda d: d["std_coef"].abs())


def residual_anomaly(target, cfg):
    tr, va, te = load_splits(target, cfg)
    alldf = pd.concat([tr, va, te]).sort_values("date").reset_index(drop=True)
    tmask = (alldf["date"] <= pd.Timestamp(cfg["split"]["train_end"])).to_numpy()
    resid = alldf["y"].to_numpy() - fit_baseline(alldf, tmask, "y")
    out = []
    for c in [c for c in alldf.columns if c.startswith("anom_")]:
        out.append({"target": target, "anomaly": c,
                    "corr_with_residual": np.corrcoef(alldf[c], resid)[0, 1]})
    return pd.DataFrame(out)


def badweather_eval(target, cfg):
    tr, va, te = load_splits(target, cfg)
    rows = []
    # 天候ショック日 = test における降水アノマリの前日差が上位 (1-quantile)
    shock_var = "anom_" + cfg["evaluation"]["weather_shock"]["var"]
    q = cfg["evaluation"]["weather_shock"]["quantile"]
    te = te.sort_values("date").copy()
    te["_shock_metric"] = te[shock_var].diff().abs()
    thr = te["_shock_metric"].quantile(q)
    shock = te["_shock_metric"] >= thr

    for condition in ("C", "D"):  # C=lags only, D=lags+weather
        feats = condition_columns(list(tr.columns), condition)
        pipe = Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=ALPHAS))]).fit(
            tr[feats], tr["y"])
        pred = pipe.predict(te[feats])
        rows.append({
            "target": target, "condition": condition,
            "mae_all": mean_absolute_error(te["y"], pred),
            "mae_shock": mean_absolute_error(te["y"][shock], pred[shock.to_numpy()]),
            "n_shock": int(shock.sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    targets = ["total", "member", "casual"] if cfg["citibike"]["keep_user_types"] else ["total"]
    rdir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    rdir.mkdir(parents=True, exist_ok=True)

    coefs = pd.concat([standardized_coefs(t, cfg) for t in targets], ignore_index=True)
    coefs.sort_values(["target", "abs"], ascending=[True, False]).to_csv(
        rdir / "standardized_coef.csv", index=False)

    ra = pd.concat([residual_anomaly(t, cfg) for t in targets], ignore_index=True)
    ra.to_csv(rdir / "residual_anomaly.csv", index=False)

    bw = pd.concat([badweather_eval(t, cfg) for t in targets], ignore_index=True)
    bw.to_csv(rdir / "badweather_eval.csv", index=False)

    print("=== standardized coef (|coef| desc, top per target) ===")
    for t in targets:
        sub = coefs[coefs["target"] == t].sort_values("abs", ascending=False).head(6)
        print(f"[{t}]"); print(sub[["feature", "group", "std_coef"]].to_string(index=False))
    print("\n=== residual vs weather anomaly corr ===")
    print(ra.to_string(index=False))
    print("\n=== bad-weather (shock-day) eval ===")
    print(bw.to_string(index=False))


if __name__ == "__main__":
    main()
