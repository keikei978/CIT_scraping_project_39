# ===== リーク/交絡検査（裁量性×天候 Phase A） =====
# 既存枠の規律を継承: baseline/anomaly は train のみで fit、暦特徴に生weather非混入、
# ランダム分割なし。

import numpy as np
import pandas as pd

from src.baseline import fit_baseline, weather_anomaly
from src.analyze_segments import _cal_features


def make_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    doy = dates.dayofyear.to_numpy()
    season = 1000 + 300 * np.sin(2 * np.pi * doy / 365.25)
    temp = 12 + 11 * np.sin(2 * np.pi * (doy - 100) / 365.25) + rng.normal(0, 2, n)
    precip = np.clip(rng.gamma(0.5, 3, n) - 1, 0, None)
    n_rides = season + 20 * (temp - temp.mean()) - 15 * precip + rng.normal(0, 50, n)
    return pd.DataFrame({"date": dates, "n": n_rides.round(),
                         "temperature_2m_mean": temp, "precipitation_sum": precip})


TRAIN_END = pd.Timestamp("2022-09-30")


def _tmask(df):
    return (df["date"] <= TRAIN_END).to_numpy()


# ---------- 気象アノマリは train のみで climatology を fit ----------
def test_weather_anomaly_fit_on_train_only():
    df = make_df()
    tm = _tmask(df)
    a1 = weather_anomaly(df, tm, "temperature_2m_mean")
    df2 = df.copy()
    df2.loc[df2.index[-10:], "temperature_2m_mean"] += 99
    a2 = weather_anomaly(df2, tm, "temperature_2m_mean")
    assert np.allclose(a1[tm], a2[tm]), "アノマリ climatology が train 外を覗いている疑い"


# ---------- ベースラインは train のみで fit ----------
def test_baseline_fit_on_train_only():
    df = make_df()
    tm = _tmask(df)
    b1 = fit_baseline(df.assign(y=df["n"]), tm, "y")
    df2 = df.copy()
    df2.loc[df2.index[-10:], "n"] += 9999
    b2 = fit_baseline(df2.assign(y=df2["n"]), tm, "y")
    assert np.allclose(b1[tm], b2[tm]), "ベースラインが test の y を覗いている疑い"


# ---------- 暦特徴に生 weather が混入していない（交絡防止） ----------
def test_no_raw_weather_in_cal_features():
    df = make_df()
    cal = _cal_features(pd.to_datetime(df["date"]))
    for c in cal.columns:
        assert c.startswith("cal_"), f"非暦特徴が混入: {c}"
        for w in ("temperature", "precipitation", "windspeed"):
            assert w not in c, f"暦特徴 {c} に生weather {w} が混入"


# ---------- ランダム分割を使っていない ----------
def test_no_random_split_in_source():
    import pathlib
    for f in pathlib.Path("src").glob("*.py"):
        text = f.read_text(encoding="utf-8")
        if "train_test_split" in text:
            assert "shuffle=False" in text, f"{f.name} でシャッフル分割の疑い"
