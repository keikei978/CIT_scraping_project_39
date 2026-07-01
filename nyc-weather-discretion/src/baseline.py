"""
baseline.py — Phase 3 の核: カレンダー・ベースライン と 気象アノマリ（どちらも train のみで fit）。

【精査を反映した設計判断（重要・PLAN §3.2 からの修正）】
PLAN 原案は「月をベースラインに入れない（生weatherが季節を盗まれるのを防ぐため）」だった。
本実装は weather を **アノマリ（平年偏差）** で投入するため、その懸念は解消される:
アノマリは「その時期の平年からのズレ」で季節と直交するので、ベースラインに季節成分を
入れても weather の効果を盗まない。むしろ季節を入れないと、需要の季節スイング（夏高・冬低）が
残差に残り、それを weather が“季節の代理”として説明してしまう（交絡）。
→ よって baseline = 曜日 + 平滑季節(調和関数) + 成長トレンド（weatherは一切使わない）。
   weather は別途 day-of-year 平年値からのアノマリにする。
両者とも day-of-year の平年構造を **train 期間のみ** で推定し、val/test には適用のみ（fit/transform分離）。

識別の論理:
  residual = demand − baseline（時期・曜日・トレンドで説明できない分）
  この residual を weather **アノマリ**で説明できるか＝「平年より暑い/雨の日に需要が動くか」。
  これは季節交絡を構造的に排した、weather の純粋効果の推定。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _design(dates: pd.Series, day_index: np.ndarray, n_harmonics: int,
            include_dow: bool, include_trend: bool) -> np.ndarray:
    """OLS 用デザイン行列: 切片 + (trend) + (dow ダミー) + 調和季節(sin/cos)。weatherは含めない。"""
    n = len(dates)
    cols = [np.ones(n)]
    if include_trend:
        cols.append(day_index.astype(float))
    if include_dow:
        dow = dates.dt.dayofweek.to_numpy()
        for d in range(1, 7):                 # drop_first（月曜=基準）
            cols.append((dow == d).astype(float))
    doy = dates.dt.dayofyear.to_numpy().astype(float)
    for k in range(1, n_harmonics + 1):       # 年周期の調和関数（平滑季節）
        cols.append(np.sin(2 * np.pi * k * doy / 365.25))
        cols.append(np.cos(2 * np.pi * k * doy / 365.25))
    return np.column_stack(cols)


def _fit_predict(y: np.ndarray, X: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """train 行のみで OLS を fit し、全行へ予測を返す（fit/transform 分離）。"""
    beta, *_ = np.linalg.lstsq(X[train_mask], y[train_mask], rcond=None)
    return X @ beta


def fit_baseline(df: pd.DataFrame, train_mask: np.ndarray, target: str,
                 n_harmonics: int = 3) -> np.ndarray:
    """
    需要のカレンダー・ベースライン（曜日 + 平滑季節 + 成長トレンド）を train のみで fit。
    weather は一切使わない。全行のベースライン配列を返す。
    """
    dates = pd.to_datetime(df["date"])
    day_index = (dates - dates.min()).dt.days.to_numpy()
    X = _design(dates, day_index, n_harmonics, include_dow=True, include_trend=True)
    return _fit_predict(df[target].to_numpy().astype(float), X, train_mask)


def weather_anomaly(df: pd.DataFrame, train_mask: np.ndarray, var: str,
                    n_harmonics: int = 3) -> np.ndarray:
    """
    気象変数の day-of-year 平年構造（調和・成長トレンドなし）を train のみで fit し、
    アノマリ = 実測 − 平年 を全行で返す。これが季節と直交する weather 特徴。
    """
    dates = pd.to_datetime(df["date"])
    day_index = (dates - dates.min()).dt.days.to_numpy()
    X = _design(dates, day_index, n_harmonics, include_dow=False, include_trend=False)
    clim = _fit_predict(df[var].to_numpy().astype(float), X, train_mask)
    return df[var].to_numpy().astype(float) - clim
