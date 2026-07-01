# ===== リーク検査（Citi Bike 残差×天候プロジェクト） =====
# タクシー版の検査に加え、本研究固有のリスク（ベースライン交絡・アノマリの train-only fit・
# 暦特徴への生weather混入）を機械的に検査する。

import numpy as np
import pandas as pd
import pytest

from src.baseline import fit_baseline, weather_anomaly
from src.features import build_features, condition_columns


def make_cfg():
    return {
        "split": {"train_end": "2022-09-30", "val_end": "2022-11-15"},
        "weather": {"daily_vars": ["temperature_2m_mean", "precipitation_sum"],
                    "anomaly": {"harmonics": 3}},
        "features": {"lags": [1, 7, 14], "rolling_windows": [7]},
    }


def make_merged(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    doy = dates.dayofyear.to_numpy()
    season = 1000 + 300 * np.sin(2 * np.pi * doy / 365.25)
    temp = 12 + 11 * np.sin(2 * np.pi * (doy - 100) / 365.25) + rng.normal(0, 2, n)
    precip = np.clip(rng.gamma(0.5, 3, n) - 1, 0, None)
    total = season + 20 * (temp - temp.mean()) - 15 * precip + rng.normal(0, 50, n)
    return pd.DataFrame({"date": dates, "total": total.round(),
                         "member": (total * 0.7).round(), "casual": (total * 0.3).round(),
                         "temperature_2m_mean": temp, "precipitation_sum": precip})


# ---------- A: ラグは正の shift のみ（未来非参照） ----------
def test_lag_uses_positive_shift():
    cfg = make_cfg()
    feats = build_features(make_merged(), "total", cfg)
    expected = feats["y"].shift(1)
    m = expected.notna()
    assert np.allclose(feats.loc[m, "lag_y_1"], expected[m]), "lag_y_1 が y.shift(1) と不一致"


# ---------- B: rolling は当日を含まない（shift(1)後） ----------
def test_rolling_excludes_current_day():
    feats = build_features(make_merged(), "total", make_cfg())
    roll = [c for c in feats.columns if c.startswith("lag_roll_")]
    assert roll, "rolling 特徴が無い"
    for c in roll:
        assert feats[c].isna().sum() > 0, f"{c} に当日混入の疑い（先頭NaNが無い）"


# ---------- C: 気象アノマリは train のみで climatology を fit ----------
def test_weather_anomaly_fit_on_train_only():
    cfg = make_cfg()
    df = make_merged()
    tmask = (df["date"] <= pd.Timestamp(cfg["split"]["train_end"])).to_numpy()
    a1 = weather_anomaly(df, tmask, "temperature_2m_mean")
    df2 = df.copy()
    df2.loc[df2.index[-10:], "temperature_2m_mean"] += 99  # test 期間を改変
    a2 = weather_anomaly(df2, tmask, "temperature_2m_mean")
    # train 行のアノマリは test 改変の影響を受けてはいけない（climatologyがtrainのみのため）
    assert np.allclose(a1[tmask], a2[tmask]), "アノマリの climatology が train 以外を覗いている疑い"


# ---------- D: ベースラインは train のみで fit ----------
def test_baseline_fit_on_train_only():
    cfg = make_cfg()
    df = make_merged()
    tmask = (df["date"] <= pd.Timestamp(cfg["split"]["train_end"])).to_numpy()
    b1 = fit_baseline(df, tmask, "total")
    df2 = df.copy()
    df2.loc[df2.index[-10:], "total"] += 9999  # test 期間の y を改変
    b2 = fit_baseline(df2, tmask, "total")
    assert np.allclose(b1[tmask], b2[tmask]), "ベースラインが test の y を覗いている疑い"


# ---------- E: ベースライン交絡防止（暦特徴に生 weather が混入していない） ----------
def test_no_raw_weather_in_calendar_features():
    cfg = make_cfg()
    feats = build_features(make_merged(), "total", cfg)
    cal = [c for c in feats.columns if c.startswith("cal_")]
    wvars = cfg["weather"]["daily_vars"]
    # 暦特徴名に生 weather 変数が混入していないこと。weather は anom_* のみで入る。
    for c in cal:
        for w in wvars:
            assert w not in c, f"暦特徴 {c} に生weather {w} が混入（交絡の疑い）"
    anom = [c for c in feats.columns if c.startswith("anom_")]
    assert len(anom) == len(wvars), "weather が anom_* として全て入っていない"
    # 条件A（暦のみ）に weather 由来列が無いこと
    assert all(not c.startswith("anom_") for c in condition_columns(list(feats.columns), "A"))


# ---------- F: ランダム分割を使っていない ----------
def test_no_random_split_in_source():
    import pathlib
    for f in pathlib.Path("src").glob("*.py"):
        text = f.read_text(encoding="utf-8")
        if "train_test_split" in text:
            assert "shuffle=False" in text, f"{f.name} でシャッフル分割の疑い"
