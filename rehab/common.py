"""
common.py — 更生作業の共有ユーティリティ。

- 自転車 baseline.py と同一のロジック（曜日+平滑季節(調和)+成長トレンドを train のみで fit、
  weather は day-of-year 平年からのアノマリ）を再実装し、3プロジェクトに同一手法を適用できるようにする。
- ブロック・ブートストラップ（自己相関を尊重）で相関・R² の信頼区間を出す関数。
- 残差の1次自己相関から有効サンプル数を出し、Fisher-z で相関差の検定を行う関数。

すべて外部依存は numpy/pandas/sklearn/holidays のみ（taxi の .venv に導入済み）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as _holidays
except Exception:  # pragma: no cover
    _holidays = None


# ---------------------------------------------------------------------------
# 自転車 baseline.py と同一のベースライン / アノマリ（train-only fit）
# ---------------------------------------------------------------------------
def _design(dates: pd.Series, day_index: np.ndarray, n_harmonics: int,
            include_dow: bool, include_trend: bool) -> np.ndarray:
    n = len(dates)
    cols = [np.ones(n)]
    if include_trend:
        cols.append(day_index.astype(float))
    if include_dow:
        dow = dates.dt.dayofweek.to_numpy()
        for d in range(1, 7):
            cols.append((dow == d).astype(float))
    doy = dates.dt.dayofyear.to_numpy().astype(float)
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * k * doy / 365.25))
        cols.append(np.cos(2 * np.pi * k * doy / 365.25))
    return np.column_stack(cols)


def _fit_predict(y: np.ndarray, X: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X[train_mask], y[train_mask], rcond=None)
    return X @ beta


def fit_baseline(df: pd.DataFrame, train_mask: np.ndarray, target: str,
                 n_harmonics: int = 3) -> np.ndarray:
    dates = pd.to_datetime(df["date"])
    day_index = (dates - dates.min()).dt.days.to_numpy()
    X = _design(dates, day_index, n_harmonics, include_dow=True, include_trend=True)
    return _fit_predict(df[target].to_numpy().astype(float), X, train_mask)


def weather_anomaly(df: pd.DataFrame, train_mask: np.ndarray, var: str,
                    n_harmonics: int = 3) -> np.ndarray:
    dates = pd.to_datetime(df["date"])
    day_index = (dates - dates.min()).dt.days.to_numpy()
    X = _design(dates, day_index, n_harmonics, include_dow=False, include_trend=False)
    clim = _fit_predict(df[var].to_numpy().astype(float), X, train_mask)
    return df[var].to_numpy().astype(float) - clim


def cal_features(dates: pd.Series, nh: int = 3) -> pd.DataFrame:
    """暦特徴（トレンド + 曜日ダミー + 調和季節 + 祝日）。生 weather は含めない。"""
    out = pd.DataFrame(index=dates.index)
    out["cal_trend"] = (dates - dates.min()).dt.days.to_numpy().astype(float)
    dow = dates.dt.dayofweek
    for d in range(1, 7):
        out[f"cal_dow_{d}"] = (dow == d).astype(float)
    doy = dates.dt.dayofyear.to_numpy().astype(float)
    for k in range(1, nh + 1):
        out[f"cal_sin_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        out[f"cal_cos_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    if _holidays is not None:
        yrs = dates.dt.year.unique().tolist()
        us = _holidays.US(years=yrs)
        out["cal_is_holiday"] = dates.dt.date.map(lambda d: int(d in us)).astype(float)
    return out


# ---------------------------------------------------------------------------
# ブロック・ブートストラップ（自己相関を尊重した CI）
# ---------------------------------------------------------------------------
def _moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """長さ n を、長さ block の移動ブロックを復元抽出して概ね n 個に組み立てる。"""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])
    return idx[:n]


def block_bootstrap_corr(x: np.ndarray, y: np.ndarray, block: int = 14,
                         n_boot: int = 5000, seed: int = 42) -> dict:
    """corr(x,y) のブロック・ブートストラップ CI（時系列順の x,y を渡すこと）。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    point = float(np.corrcoef(x, y)[0, 1])
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = _moving_block_indices(n, block, rng)
        xb, yb = x[idx], y[idx]
        boots[b] = np.corrcoef(xb, yb)[0, 1] if np.std(xb) > 0 and np.std(yb) > 0 else np.nan
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_excludes_0": bool(lo > 0 or hi < 0)}


def block_bootstrap_stat(values: np.ndarray, statfn, block: int = 14,
                         n_boot: int = 2000, seed: int = 42) -> dict:
    """任意統計量のブロック・ブートストラップ CI。values は時系列順の行インデックス基盤。"""
    n = len(values)
    rng = np.random.default_rng(seed)
    point = float(statfn(np.arange(n)))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = _moving_block_indices(n, block, rng)
        boots[b] = statfn(idx)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi)}


# ---------------------------------------------------------------------------
# 有効サンプル数（自己相関補正）と相関差の Fisher-z 検定
# ---------------------------------------------------------------------------
def lag1_autocorr(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = np.sum(x * x)
    return float(np.sum(x[1:] * x[:-1]) / denom) if denom > 0 else 0.0


def effective_n(resid: np.ndarray) -> float:
    """1次自己相関 r による有効 n: n_eff = n * (1-r)/(1+r)。"""
    n = len(resid)
    r = lag1_autocorr(resid)
    r = min(max(r, 0.0), 0.99)
    return n * (1 - r) / (1 + r)


def fisher_z_diff(corr_a: float, n_a: float, corr_b: float, n_b: float) -> dict:
    """独立2群の相関差の Fisher-z 検定（有効 n を渡せば自己相関補正済み）。"""
    def z(r): return np.arctanh(np.clip(r, -0.999, 0.999))
    se = np.sqrt(1.0 / (n_a - 3) + 1.0 / (n_b - 3))
    zdiff = (z(corr_a) - z(corr_b)) / se
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(zdiff) / sqrt(2))))
    return {"z": float(zdiff), "p": float(p), "se": float(se)}
