"""
merge.py - daily_demand に weather と events を date で left join して統合

処理:
  - daily_demand を左テーブルに weather / events を date で left join
  - weather の欠損は join 後に報告（archive-api は基本全日揃う想定）
  - events の欠損は 0 で埋める（イベントが無い日 = 0 件）
  - assert: 行数が daily_demand と同一、date×zone ユニーク性を確認
  - data/merged.parquet に保存
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config
from src.weather import fetch_weather
from src.events import fetch_events, fetch_events_by_zone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def merge_all(
    daily_demand: pd.DataFrame,
    weather_df: pd.DataFrame,
    events_df: pd.DataFrame,
    zone_events_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    daily_demand に weather と events を left join して統合 DataFrame を返す。

    Parameters
    ----------
    daily_demand   : DataFrame with columns [date, zone, demand]
    weather_df     : DataFrame with columns [date, ...weather vars...]
    events_df      : DataFrame with columns [date, event_count]
                     （全市スカラー。条件 C/D 用。date でのみ結合＝全ゾーン同値）
    zone_events_df : DataFrame with columns [date, zone, event_count] or None
                     （ゾーン解像。条件 E 用。(date, zone) で結合。None なら列を作らない）

    Returns
    -------
    merged DataFrame (行数は daily_demand と同一)
    """
    n_expected = len(daily_demand)
    logger.info("Merging: daily_demand shape=%s", daily_demand.shape)

    # date 型を統一
    daily_demand = daily_demand.copy()
    daily_demand["date"] = pd.to_datetime(daily_demand["date"]).dt.normalize()

    weather_df = weather_df.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.normalize()

    events_df = events_df.copy()
    events_df["date"] = pd.to_datetime(events_df["date"]).dt.normalize()

    # weather を left join
    merged = daily_demand.merge(weather_df, on="date", how="left")
    logger.info("After weather join: shape=%s", merged.shape)

    # weather の欠損を報告
    weather_cols = [c for c in weather_df.columns if c != "date"]
    weather_missing = merged[weather_cols].isnull().sum()
    weather_missing = weather_missing[weather_missing > 0]
    if not weather_missing.empty:
        logger.warning(
            "Missing values in weather columns after join:\n%s",
            weather_missing.to_string(),
        )
    else:
        logger.info("No missing values in weather columns after join.")

    # events を left join
    merged = merged.merge(events_df, on="date", how="left")
    logger.info("After events join: shape=%s", merged.shape)

    # events の欠損は 0 で埋める（イベント無し = 0 件）
    if "event_count" in merged.columns:
        n_filled = merged["event_count"].isnull().sum()
        if n_filled > 0:
            logger.info("Filling %d NaN in event_count with 0", n_filled)
        merged["event_count"] = merged["event_count"].fillna(0).astype(int)

    # ゾーン解像イベント（条件 E 用）を (date, zone) で left join。
    # 列名を event_count_zone にして全市スカラー event_count と区別する。
    if zone_events_df is not None:
        zone_events_df = zone_events_df.copy()
        zone_events_df["date"] = pd.to_datetime(zone_events_df["date"]).dt.normalize()
        zone_events_df = zone_events_df.rename(columns={"event_count": "event_count_zone"})
        # 結合キー zone の dtype を明示的に揃える（不一致だと全行マッチせず
        # event_count_zone が全ゼロ化＝条件 E が A へサイレント退化する経路を塞ぐ）。
        merged["zone"] = merged["zone"].astype(int)
        zone_events_df["zone"] = zone_events_df["zone"].astype(int)
        merged = merged.merge(zone_events_df, on=["date", "zone"], how="left")
        logger.info("After zone-events join: shape=%s", merged.shape)
        # 欠損 (date, zone) = そのゾーン・その日にイベント無し = 真の 0。
        n_filled = merged["event_count_zone"].isnull().sum()
        if n_filled > 0:
            logger.info("Filling %d NaN in event_count_zone with 0", n_filled)
        merged["event_count_zone"] = merged["event_count_zone"].fillna(0.0).astype(float)

        # 統制条件 C'（Manhattan限定・空間集約スカラー）用の列。
        # event_count_zone を日付で全ゾーン合算し、同一日は全ゾーン同値にする。
        # これは E とまったく同じ基礎データ（同じ Manhattan イベント・同じ面積按分・
        # 同じ連続単位）を「ゾーン解像」ではなく「日次スカラー」に集約したもの。
        # E vs C' で空間解像度の効果のみを分離でき、C vs E の交絡（区スコープ＋
        # 単位の違い）を解消する（leak-reviewer 重大-1 への対応）。
        merged["event_count_manhattan"] = merged.groupby("date")[
            "event_count_zone"
        ].transform("sum")

    # assert: 行数が daily_demand と同一
    assert len(merged) == n_expected, (
        f"Row count changed after merge: expected {n_expected}, got {len(merged)}. "
        "Check for duplicate keys in weather or events DataFrames."
    )

    # assert: date×zone ユニーク性
    dup_count = merged[["date", "zone"]].duplicated().sum()
    assert dup_count == 0, (
        f"Found {dup_count} duplicate (date, zone) pairs after merge."
    )

    logger.info(
        "Merge assertions passed: %d rows, %d columns, "
        "all (date,zone) pairs unique.",
        len(merged),
        len(merged.columns),
    )

    return merged


def load_daily_demand(path: str | Path) -> pd.DataFrame:
    """保存済みの daily_demand.parquet を読み込む。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"daily_demand not found: {p}")
    df = pd.read_parquet(p)
    logger.info("Loaded daily_demand from %s  shape=%s", p, df.shape)
    return df


def save_merged(df: pd.DataFrame, path: str | Path) -> None:
    """merged DataFrame を Parquet に保存する。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    logger.info("Saved merged to %s  shape=%s", p, df.shape)


def run_merge(cfg: dict[str, Any], project_root: Path) -> pd.DataFrame:
    """
    cfg に従って merge を実行し、保存した DataFrame を返す。
    """
    raw_path = project_root / cfg["paths"]["raw_demand"]
    merged_path = project_root / cfg["paths"]["merged"]

    daily_demand = load_daily_demand(raw_path)
    weather_df = fetch_weather(cfg)
    events_df = fetch_events(cfg)
    zone_events_df = fetch_events_by_zone(cfg)

    merged = merge_all(daily_demand, weather_df, events_df, zone_events_df)
    save_merged(merged, merged_path)

    return merged


if __name__ == "__main__":
    cfg = load_config()
    project_root = Path(__file__).resolve().parent.parent

    merged = run_merge(cfg, project_root)

    print("\n=== merged summary ===")
    print(f"shape   : {merged.shape}")
    print(f"columns : {merged.columns.tolist()}")
    print(f"dtypes  :\n{merged.dtypes.to_string()}")
    print("\nHead:")
    print(merged.head(5).to_string(index=False))

    # 欠損サマリ
    missing_summary = merged.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    if not missing_summary.empty:
        print(f"\nRemaining missing values:\n{missing_summary.to_string()}")
    else:
        print("\nNo missing values in merged data.")
