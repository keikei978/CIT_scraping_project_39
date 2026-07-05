"""
ステップ1（D1を潰す）: タクシーに自転車と同一の手法を当て、対比が本物か確認する。

問い: 「タクシーは天候が効かない／自転車は効く」は【モードの差】か、それとも
      【手法の差（タクシーは生weather+予測R²、自転車はアノマリ+残差化）】のアーティファクトか。

方法: 自転車 baseline.py と同一のロジック（train-only の残差化＋day-of-year アノマリ）を、
      タクシーの【日次システム全体需要】（ゾーン合算＝自転車と同じ集計粒度）に適用し、
      同じコードで自転車 total と並べる:
        (1) 残差 × 降水/気温アノマリ の相関（自転車の residual_anomaly と同形）
        (2) A(暦)→B(暦+天候) の test 増分 R²（線形）
        (3) 標準化係数（暦+天候アノマリ+ラグ）で天候アノマリが上位に来るか

判定: タクシーでも相関が強く負／増分R²が大きければ「対比は手法アーティファクト」。
      タクシーは0近傍のままなら「対比は本物（モード差）」。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fit_baseline, weather_anomaly, cal_features

BASE = Path("C:/Users/keisu/Documents/projects/CIT_Assignment/scraping_project")
ALPHAS = np.logspace(-2, 4, 30)
COMMON_WVARS = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "windspeed_10m_max"]
NH = 3


def load_taxi_daily_total() -> pd.DataFrame:
    d = pd.read_parquet(BASE / "nyc-taxi-demand/data/merged.parquet")
    d["date"] = pd.to_datetime(d["date"])
    agg = d.groupby("date").agg(
        y=("demand", "sum"),
        **{v: (v, "first") for v in COMMON_WVARS}
    ).reset_index().sort_values("date").reset_index(drop=True)
    return agg


def load_citibike_daily_total() -> pd.DataFrame:
    d = pd.read_parquet(BASE / "nyc-citibike-demand/data/merged.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d = d.rename(columns={"total": "y"}).sort_values("date").reset_index(drop=True)
    return d[["date", "y"] + COMMON_WVARS]


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for L in (1, 7, 14):
        df[f"lag_y_{L}"] = df["y"].shift(L)
    df["lag_roll_mean_7"] = df["y"].shift(1).rolling(7).mean()
    return df


def analyze(df: pd.DataFrame, label: str, train_end: str, val_end: str) -> dict:
    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    train_mask = (dates <= pd.Timestamp(train_end)).to_numpy()
    te = (dates > pd.Timestamp(val_end)).to_numpy()
    tr = train_mask

    # (1) 残差 × アノマリ相関（全期間: 自転車 analyze.py と同形）
    resid = df["y"].to_numpy().astype(float) - fit_baseline(df, train_mask, "y", NH)
    anoms = {v: weather_anomaly(df, train_mask, v, NH) for v in COMMON_WVARS}
    corr = {v: float(np.corrcoef(anoms[v], resid)[0, 1]) for v in COMMON_WVARS}

    # (2) A→B 増分 R²（線形, test）
    cal = cal_features(dates, NH)
    anom_df = pd.DataFrame({f"anom_{v}": anoms[v] for v in COMMON_WVARS}, index=df.index)

    def r2_test(X):
        m = Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=ALPHAS))]).fit(X[tr], df["y"].to_numpy()[tr])
        return r2_score(df["y"].to_numpy()[te], m.predict(X[te]))

    r2A = r2_test(cal.to_numpy())
    r2B = r2_test(np.column_stack([cal.to_numpy(), anom_df.to_numpy()]))

    # (3) 標準化係数（暦+天候アノマリ+ラグ, train のみ, RidgeCV）
    dl = add_lags(df.assign(**{f"anom_{v}": anoms[v] for v in COMMON_WVARS}))
    lag_cols = ["lag_y_1", "lag_y_7", "lag_y_14", "lag_roll_mean_7"]
    feat_df = pd.concat([cal, anom_df, dl[lag_cols]], axis=1)
    keep = dl[lag_cols].notna().all(axis=1).to_numpy()
    tr2 = tr & keep
    pipe = Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=ALPHAS))]).fit(
        feat_df[tr2], df["y"].to_numpy()[tr2])
    coefs = pd.Series(pipe.named_steps["m"].coef_, index=feat_df.columns)
    grp = {}
    for c in feat_df.columns:
        g = "weather_anom" if c.startswith("anom_") else ("lag" if c.startswith("lag_") else "calendar")
        grp.setdefault(g, 0.0)
        grp[g] = max(grp[g], abs(coefs[c]))
    top_precip = coefs.get("anom_precipitation_sum", np.nan)
    top_lag1 = coefs.get("lag_y_1", np.nan)

    return {"label": label, "n_days": int(len(df)),
            "corr": corr, "r2A": r2A, "r2B": r2B, "dR2": r2B - r2A,
            "grp_max_abs": grp, "coef_precip": float(top_precip), "coef_lag1": float(top_lag1)}


def main():
    taxi = load_taxi_daily_total()
    cbk = load_citibike_daily_total()

    results = [
        analyze(cbk, "CITIBIKE total (2022-2023)", "2023-06-30", "2023-09-30"),
        # タクシーを自転車と同一期間(2022-2023)へ制限し、同一splitで（最も公平なD1比較）
        analyze(taxi[pd.to_datetime(taxi["date"]) <= "2023-12-31"].reset_index(drop=True),
                "TAXI total (2022-2023, bike-split)", "2023-06-30", "2023-09-30"),
        # タクシー全期間（本来のsplit）も参考に
        analyze(taxi, "TAXI total (2022-2024, own-split)", "2023-12-31", "2024-06-30"),
    ]

    print("\n" + "=" * 78)
    print("ステップ1: 同一手法（残差化+アノマリ）での タクシー vs 自転車 対比")
    print("=" * 78)

    print("\n--- (1) 残差 × 天候アノマリ の相関（負ほど雨に弱い / 正ほど暖かさで増）---")
    hdr = f"{'系列':40s} {'precip_sum':>11s} {'temp_max':>9s} {'wind_max':>9s}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['label']:40s} {r['corr']['precipitation_sum']:+11.3f} "
              f"{r['corr']['temperature_2m_max']:+9.3f} {r['corr']['windspeed_10m_max']:+9.3f}")

    print("\n--- (2) A(暦)→B(暦+天候) の test 増分 R²（線形）---")
    hdr = f"{'系列':40s} {'r2A':>8s} {'r2B':>8s} {'ΔR2':>8s}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['label']:40s} {r['r2A']:8.3f} {r['r2B']:8.3f} {r['dR2']:+8.3f}")

    print("\n--- (3) 標準化係数: グループ別 最大|coef| と 降水/ラグ1 の係数 ---")
    hdr = f"{'系列':40s} {'cal':>9s} {'weather':>9s} {'lag':>9s} | {'precip':>9s} {'lag1':>9s}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        g = r["grp_max_abs"]
        print(f"{r['label']:40s} {g.get('calendar',0):9.0f} {g.get('weather_anom',0):9.0f} "
              f"{g.get('lag',0):9.0f} | {r['coef_precip']:+9.0f} {r['coef_lag1']:+9.0f}")

    print("\n判定の読み方: タクシーでも precip 相関が強く負 & ΔR2 が大きければ【手法アーティファクト】。")
    print("            タクシーが 0 近傍のままなら【対比は本物＝モード差】。")


if __name__ == "__main__":
    main()
