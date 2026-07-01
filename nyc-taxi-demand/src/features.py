"""
features.py — Phase 3: 特徴量生成 + 4条件データセット作成

リーク防止方針の概要:
  - demand 由来のラグ/rolling は正の shift のみ（過去値のみ参照、未来参照禁止）。
  - rolling は必ず shift(1) 後に計算し、当日の demand を含めない。
  - ゾーンをまたいだ shift/rolling を防ぐため、必ず groupby('zone') する。
  - 気象・イベント・カレンダーは「予測対象日に予測時点で入手可能な外生情報」として
    同日（shift しない）値を使う。カレンダーは確定情報、気象は数値予報で事前に既知、
    許可イベントは事前申請済みのため、これらを同日値として使ってもターゲット demand の
    未来値を参照することにはならず、情報リークに該当しない。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import holidays
import numpy as np
import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)


# ===========================
# 特徴量構築コア関数
# ===========================

def build_features(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    入力 DataFrame に特徴量を付与して返す。

    Parameters
    ----------
    df : pd.DataFrame
        最低限 ['date', 'zone', 'demand'] の3列を含む DataFrame。
        気象列・イベント列は任意（あれば保持される）。
    lags : list of int, optional
        ラグ日数リスト。省略時は config の features.lags を使用。
    rolling_windows : list of int, optional
        rolling window 日数リスト。省略時は config の features.rolling_windows を使用。

    Returns
    -------
    pd.DataFrame
        (zone, date) 昇順でソートされた特徴量付き DataFrame。
        ラグ/rolling 起因の NaN は残す（dropna しない）。
    """
    cfg = load_config()
    feat_cfg = cfg["features"]

    if lags is None:
        lags = feat_cfg["lags"]
    if rolling_windows is None:
        rolling_windows = feat_cfg["rolling_windows"]

    # --- 必須列の確認 ---
    required_cols = {"date", "zone", "demand"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"必須列が不足しています: {missing}")

    # --- ソート: (zone, date) 昇順 ---
    # groupby('zone').shift(lag) は行の並び順に依存するため、
    # (zone, date) 昇順にソートしておく必要がある。
    # テストも out 上で shift を再計算して比較するため、同じ並び順が必要。
    df = df.sort_values(["zone", "date"]).reset_index(drop=True)

    out = df.copy()

    # ===========================
    # (A) ラグ特徴量
    # ===========================
    # リーク防止: groupby('zone')['demand'].shift(lag) は
    # 各ゾーン内で lag 日前の demand 値を参照する。
    # shift(lag) で lag >= 1 の正の値を使うため、未来参照にはならない。
    # ゾーンをまたいだ shift を防ぐため必ず groupby('zone') を使用。
    for lag in lags:
        col_name = f"demand_lag_{lag}"
        # groupby('zone') でゾーン境界をまたがないようにしつつ、
        # shift(lag) で lag 日前の値を取得（正の shift = 過去参照のみ）
        out[col_name] = out.groupby("zone")["demand"].shift(lag)
        # NaN は各ゾーン先頭 lag 行に残る（正常動作、dropna しない）

    # ===========================
    # (B) Rolling 特徴量
    # ===========================
    # リーク防止: まず shift(1) で1日前の demand にずらし、
    # その後 rolling(window).mean() を計算する。
    # これにより当日の demand が rolling 計算に含まれない（当日値を含まない）。
    # ゾーンをまたいだ計算を防ぐため、shift と rolling を groupby('zone').transform
    # の中で完結させる。こうすることで rolling のウィンドウがゾーン境界を絶対に
    # またがない（min_periods のデフォルト挙動に依存せず、構造的にリーク不能）。
    for window in rolling_windows:
        col_name = f"demand_roll_mean_{window}"
        # transform 内でゾーンごとに shift(1)→rolling(window) を閉じて計算する。
        # → 各ゾーンの先頭 (window) 行は NaN になる（正常動作）
        out[col_name] = out.groupby("zone")["demand"].transform(
            lambda s: s.shift(1).rolling(window=window).mean()
        )

    # ===========================
    # (C) カレンダー/祝日特徴量
    # ===========================
    # リーク防止: 以下の特徴量はすべて date 列（予測対象日の日付情報）から決まる。
    # 曜日・月・祝日フラグ・週末フラグは予測時点で確定している情報であり、
    # demand の未来値を参照するリークには該当しない。

    dates = out["date"]

    # --- 曜日の sin/cos エンコーディング ---
    # リーク防止: dayofweek は date から決定論的に計算される確定情報。
    # sin/cos で周期性を表現（月曜=0, 日曜=6 の循環構造を保持）。
    dow = dates.dt.dayofweek  # 0=月曜, 6=日曜
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # --- 月の sin/cos エンコーディング ---
    # リーク防止: month は date から決定論的に計算される確定情報。
    # sin/cos で12ヶ月の周期性を表現（12月と1月の連続性を保持）。
    month = dates.dt.month  # 1〜12
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)

    # --- 祝日フラグ ---
    # リーク防止: 米国祝日は事前に確定しているカレンダー情報。
    # holidays.US でフラグ化する。予測時点で既知の情報であり、
    # demand の未来値を参照するリークには該当しない。
    years = dates.dt.year.unique().tolist()
    us_holidays = holidays.US(years=years)
    out["is_holiday"] = dates.dt.date.apply(lambda d: int(d in us_holidays))

    # --- 週末フラグ ---
    # リーク防止: dayofweek から決まる確定情報（土=5, 日=6）。
    out["is_weekend"] = (dow >= 5).astype(int)

    return out


# ===========================
# 4条件データセット作成
# ===========================

WEATHER_COLS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "snowfall_sum",
    "windspeed_10m_max",
]
EVENT_COLS = ["event_count"]
# ゾーン解像イベント（条件 E 用）。全市スカラー event_count とは別列。
ZONE_EVENT_COLS = ["event_count_zone"]
# Manhattan限定・空間集約スカラー（統制条件 C' 用）。E と同一データを日次合算したもの。
MANHATTAN_EVENT_COLS = ["event_count_manhattan"]


def create_ablation_datasets(cfg: Optional[dict] = None) -> dict[str, pd.DataFrame]:
    """
    merged.parquet からアブレーション用データセットを作成する。

    条件:
      A : 時間のみ           — date, zone, demand + ラグ + rolling + カレンダー/祝日
      B : +気象              — A + 気象5列
      C : +イベント(全市)    — A + event_count（全5区スカラー＝同一日は全ゾーン同値）
      Cprime: +イベント(Manhattanスカラー)
                             — A + event_count_manhattan（E と同一データの日次合算スカラー）
      D : 全部               — A + 気象5列 + event_count
      E : +イベント(ゾーン解像) — A + event_count_zone（(date,zone)別＝ゾーンで異なる）

    対比の設計:
      - E vs Cprime : 空間解像度の効果のみを純粋に分離（同一データ・同一単位）。
      - Cprime vs C : 区スコープ（全市 vs Manhattan限定）の効果を分離。
      これにより C vs E の交絡（空間解像度＋区スコープ＋単位の混在）を解消する。

    Returns
    -------
    dict
        {'A', 'B', 'C', 'Cprime', 'D', 'E'} のうち、対応列が merged に存在するもの。
    """
    if cfg is None:
        cfg = load_config()

    merged_path = Path(cfg["paths"]["merged"])
    if not merged_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        merged_path = project_root / merged_path

    logger.info(f"merged.parquet を読み込み中: {merged_path}")
    df_raw = pd.read_parquet(merged_path)
    logger.info(f"読み込み完了: shape={df_raw.shape}")

    # --- 特徴量生成（1度だけ実行） ---
    # build_features は date, zone, demand の3列だけでも動作するが、
    # 気象・イベント列を渡して保持させ、後で条件ごとに列を出し分ける。
    df_all = build_features(df_raw)

    # --- 4条件の列セット定義 ---
    # base_cols: すべての条件に共通（A の列セット）
    # 気象・イベント列以外の列を base_cols とする
    weather_present = [c for c in WEATHER_COLS if c in df_all.columns]
    event_present = [c for c in EVENT_COLS if c in df_all.columns]
    zone_event_present = [c for c in ZONE_EVENT_COLS if c in df_all.columns]
    manhattan_event_present = [c for c in MANHATTAN_EVENT_COLS if c in df_all.columns]

    exclude_from_base = set(
        weather_present + event_present + zone_event_present + manhattan_event_present
    )
    base_cols = [c for c in df_all.columns if c not in exclude_from_base]

    datasets = {
        "A": df_all[base_cols].copy(),
        "B": df_all[base_cols + weather_present].copy(),
        "C": df_all[base_cols + event_present].copy(),
        "D": df_all[base_cols + weather_present + event_present].copy(),
    }
    # 統制条件 C'（Manhattanスカラー）とゾーン解像 E は、対応列が存在するときのみ生成。
    if manhattan_event_present:
        datasets["Cprime"] = df_all[base_cols + manhattan_event_present].copy()
    if zone_event_present:
        datasets["E"] = df_all[base_cols + zone_event_present].copy()

    # --- assert: 全条件で行数・(date,zone) が一致 ---
    n_rows_A = len(datasets["A"])
    for label, ds in datasets.items():
        assert len(ds) == n_rows_A, (
            f"条件 {label} の行数 ({len(ds)}) が A ({n_rows_A}) と不一致"
        )
        dup = ds.duplicated(subset=["date", "zone"]).sum()
        assert dup == 0, (
            f"条件 {label} に (date,zone) の重複が {dup} 行あります"
        )
        assert "demand" in ds.columns, f"条件 {label} に demand 列がありません"

    # --- 包含関係の確認: A ⊂ (B, C, D, E) ---
    set_A = set(datasets["A"].columns)
    for label in datasets:
        if label == "A":
            continue
        assert set_A <= set(datasets[label].columns), (
            f"条件 {label} が条件 A の列を包含していません"
        )

    return datasets


def save_ablation_datasets(cfg: Optional[dict] = None) -> None:
    """
    4条件データセットを parquet として保存する。

    出力先: config の paths.features_dir（デフォルト: data/features/）
    ファイル名: dataset_A.parquet, dataset_B.parquet, dataset_C.parquet, dataset_D.parquet
    """
    if cfg is None:
        cfg = load_config()

    features_dir = Path(cfg["paths"]["features_dir"])
    if not features_dir.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        features_dir = project_root / features_dir

    features_dir.mkdir(parents=True, exist_ok=True)

    datasets = create_ablation_datasets(cfg)

    for label, ds in datasets.items():
        out_path = features_dir / f"dataset_{label}.parquet"
        ds.to_parquet(out_path, index=False)
        logger.info(f"保存完了: {out_path} — shape={ds.shape}")
        logger.info(f"  列: {list(ds.columns)}")
        nan_counts = ds.isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            logger.info(f"  NaN 列: {nan_cols.to_dict()}")
        else:
            logger.info("  NaN: なし")

    print("\n=== アブレーション用データセット 生成完了 ===")
    for label, ds in datasets.items():
        nan_total = ds.isnull().sum().sum()
        print(f"  dataset_{label}.parquet: shape={ds.shape}, NaN合計={nan_total}")
        print(f"    列: {list(ds.columns)}")


# ===========================
# エントリーポイント
# ===========================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    save_ablation_datasets()
