"""
analyze_segments.py — Phase A: セグメント別の天候感応性を算出し、裁量プロキシ間で比較する。

各セグメント（自転車内・曝露と代替の向きは固定）について:
  - baseline(train のみ fit) = 曜日 + 平滑季節(調和) + 成長トレンド → 残差 r = y - baseline
  - weather アノマリ(train のみ fit)
  - 主指標 W = corr(残差, 降水アノマリ)  … スケール不変・非季節=クリーン・負ほど雨に弱い
  - 副指標 = 線形 A(暦) → B(暦+天候) の test 増分 R²

裁量プロキシ3軸で「レジャー側ほど雨に弱い（W がより負）」が一致するかを見る:
  軸1 rt_round(レジャー) vs rt_point(通勤)
  軸2 time_weMID(レジャー) vs time_wdAM(通勤)
  軸3 user_casual(レジャー) vs user_member(通勤)
"""

from __future__ import annotations

import logging
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.baseline import fit_baseline, weather_anomaly

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALPHAS = np.logspace(-2, 4, 30)


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cal_features(dates: pd.Series, nh: int = 3) -> pd.DataFrame:
    out = pd.DataFrame(index=dates.index)
    out["cal_trend"] = (dates - dates.min()).dt.days.to_numpy().astype(float)
    dow = dates.dt.dayofweek
    for d in range(1, 7):
        out[f"cal_dow_{d}"] = (dow == d).astype(float)
    doy = dates.dt.dayofyear.to_numpy().astype(float)
    for k in range(1, nh + 1):
        out[f"cal_sin_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        out[f"cal_cos_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    yrs = dates.dt.year.unique().tolist()
    us = holidays.US(years=yrs)
    out["cal_is_holiday"] = dates.dt.date.map(lambda d: int(d in us)).astype(float)
    return out


def analyze_segment(sub: pd.DataFrame, cfg: dict) -> dict:
    sub = sub.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(sub["date"])
    y = sub["n"].to_numpy().astype(float)
    train_mask = (dates <= pd.Timestamp(cfg["split"]["train_end"])).to_numpy()
    wvars = cfg["weather"]["daily_vars"]
    nh = cfg["baseline"]["harmonics"]

    # 残差と天候アノマリ
    resid = y - fit_baseline(sub.assign(y=y), train_mask, "y", n_harmonics=nh)
    anoms = {v: weather_anomaly(sub, train_mask, v, n_harmonics=nh) for v in wvars}

    res = {"segment": sub["segment"].iloc[0], "n_days": len(sub),
           "mean_per_day": int(y.mean())}
    for v in ("precipitation_hours", "precipitation_sum", "temperature_2m_max"):
        res[f"corr_{v}"] = float(np.corrcoef(anoms[v], resid)[0, 1])

    # A→B 増分R²（線形, test）
    cal = _cal_features(dates, nh)
    anom_df = pd.DataFrame({f"anom_{v}": anoms[v] for v in wvars}, index=sub.index)
    val_end = pd.Timestamp(cfg["split"]["val_end"])
    tr = train_mask
    te = (dates > val_end).to_numpy()

    def r2_test(X):
        m = Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=ALPHAS))]).fit(X[tr], y[tr])
        return r2_score(y[te], m.predict(X[te]))

    r2A = r2_test(cal.to_numpy())
    r2B = r2_test(np.column_stack([cal.to_numpy(), anom_df.to_numpy()]))
    res["r2A_test"] = round(r2A, 3)
    res["r2B_test"] = round(r2B, 3)
    res["delta_r2"] = round(r2B - r2A, 3)
    return res


def main() -> None:
    cfg = load_config()
    merged = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["merged"])
    merged["date"] = pd.to_datetime(merged["date"])

    rows = [analyze_segment(merged[merged["segment"] == s], cfg)
            for s in sorted(merged["segment"].unique())]
    df = pd.DataFrame(rows)
    for c in [c for c in df.columns if c.startswith("corr_")]:
        df[c] = df[c].round(3)

    dest = PROJECT_ROOT / cfg["paths"]["results_dir"] / "segment_weather_sensitivity.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    logger.info("Saved %s", dest)

    print(df.to_string(index=False))
    print("\n=== 裁量プロキシ対比（W=corr(残差,降水時間アノマリ), 負ほど雨に弱い） ===")
    W = df.set_index("segment")["corr_precipitation_hours"]
    pairs = [("軸1 ラウンドトリップ", "rt_round", "rt_point"),
             ("軸2 時間帯×曜日", "time_weMID", "time_wdAM"),
             ("軸3 会員種別", "user_casual", "user_member")]
    for label, leisure, commute in pairs:
        if leisure in W.index and commute in W.index:
            # 端末が cp932 の環境でも落ちないよう、判定は cp932 で表現可能な文字のみ使う
            verdict = "支持: レジャー側が雨に弱い" if W[leisure] < W[commute] else "反対: 通勤側が雨に弱い"
            print(f"  {label}: レジャー({leisure})={W[leisure]:+.3f}  vs  "
                  f"通勤({commute})={W[commute]:+.3f}   {verdict}")


if __name__ == "__main__":
    main()
