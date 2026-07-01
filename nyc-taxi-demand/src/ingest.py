"""
ingest.py - NYC TLC yellow taxi データの取得・日次×ゾーン集計

処理フロー:
  1. taxi_zone_lookup.csv から Manhattan ゾーンを特定
  2. 月別 Parquet を 1 ヶ月ずつ読み込み→即集計（生レコードはメモリ保持しない）
  3. period 内全日付 × 選択ゾーンの完全グリッドを作成し demand=0 で埋める
  4. assert で行数・欠損・非負を確認
  5. data/daily_demand.parquet に保存
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from src.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"


# ---------------------------------------------------------------------------
# ゾーン選択
# ---------------------------------------------------------------------------

def fetch_manhattan_zones(exclude_ids: list[int]) -> list[int]:
    """
    taxi_zone_lookup.csv を取得し、Borough == Manhattan の LocationID を返す。
    exclude_ids に含まれるゾーンは除外する。
    """
    logger.info("Fetching taxi zone lookup from %s", ZONE_LOOKUP_URL)
    resp = requests.get(ZONE_LOOKUP_URL, timeout=60)
    resp.raise_for_status()

    zones_df = pd.read_csv(io.StringIO(resp.text))
    # カラム名を正規化（大文字小文字・スペース対応）
    zones_df.columns = zones_df.columns.str.strip()

    manhattan = zones_df[zones_df["Borough"] == "Manhattan"]["LocationID"].tolist()
    manhattan_filtered = [z for z in manhattan if z not in exclude_ids]
    logger.info(
        "Manhattan zones: %d total, %d after excluding %s",
        len(manhattan),
        len(manhattan_filtered),
        exclude_ids,
    )
    return manhattan_filtered


# ---------------------------------------------------------------------------
# 月別集計
# ---------------------------------------------------------------------------

def _fetch_month_demand(
    year: int,
    month: int,
    manhattan_zones: list[int],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    retries: int = 4,
) -> pd.DataFrame:
    """
    1 ヶ月分の Parquet を取得し、date×zone の件数集計 DataFrame を返す。
    生レコードはこの関数スコープ内でのみ保持し、呼び出し元には集計結果だけ返す。

    ダウンロード＋パースは指数バックオフ付きでリトライする（CloudFront の
    一時的なレート制限・接続リセットは一時的なことが多いため）。全リトライ
    失敗時は RuntimeError を送出する。月をサイレントにスキップすると、その月が
    グリッド上で全 0 埋めされてデータにバイアスが入るため（leak-reviewer の
    軽微指摘）、絶対にスキップしない。

    Returns
    -------
    DataFrame with columns: date (datetime64[ns]), zone (int), demand (int)
    対象レコードが無い月は空 DataFrame を返す（=取得は成功）。
    """
    url = TLC_BASE_URL.format(year=year, month=month)
    logger.info("Fetching %s", url)

    # ダウンロード（resp.content）とパースをまとめてリトライで包む。
    # 以前は resp.content が try の外にあり、接続リセットでクラッシュしていた。
    table = None
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            raw_bytes = resp.content  # ここで実ダウンロード（必ず try 内）
            table = pq.read_table(
                pa.BufferReader(raw_bytes),
                columns=["tpep_pickup_datetime", "PULocationID"],
            )
            del raw_bytes  # 生バイトを即解放
            break
        except (requests.RequestException, pa.ArrowInvalid, OSError) as e:
            last_err = e
            wait = 5 * (2 ** (attempt - 1))  # 5, 10, 20, 40 秒
            logger.warning(
                "  attempt %d/%d failed for %s: %s%s",
                attempt,
                retries,
                url,
                e,
                f" (retrying in {wait}s)" if attempt < retries else "",
            )
            if attempt < retries:
                time.sleep(wait)

    if table is None:
        raise RuntimeError(
            f"TLC fetch/parse failed for {year}-{month:02d} after {retries} "
            f"attempts: {last_err}. Not skipping silently (would zero-fill the "
            "month and bias the grid)."
        )

    df = table.to_pandas()
    del table  # Arrow テーブルも解放

    # 日付変換（タイムゾーンを除去して date のみ）
    df["date"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce").dt.normalize()
    df = df.drop(columns=["tpep_pickup_datetime"])

    # period 外の日付を除去
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    # Manhattan ゾーンに絞る
    df = df[df["PULocationID"].isin(manhattan_zones)]

    if df.empty:
        logger.info("  -> No valid records for %d-%02d after filtering", year, month)
        return pd.DataFrame(columns=["date", "zone", "demand"])

    # 日次×ゾーンで集計
    agg = (
        df.groupby(["date", "PULocationID"], observed=True)
        .size()
        .reset_index(name="demand")
    )
    agg = agg.rename(columns={"PULocationID": "zone"})
    del df  # 生レコードを解放

    logger.info("  -> %d records aggregated for %d-%02d", len(agg), year, month)
    return agg


# ---------------------------------------------------------------------------
# トップ N ゾーン選択
# ---------------------------------------------------------------------------

def select_top_zones(
    monthly_aggs: list[pd.DataFrame],
    manhattan_zones: list[int],
    top_n: int,
    train_end: pd.Timestamp,
) -> list[int]:
    """
    月別集計リストから **train 期間（date <= train_end）のみ** の総需要を集計し、
    上位 top_n のゾーンを選ぶ。

    ゾーン集合の決定に val/test 期間の需要を使わないことで、前処理段階での
    リーク（テスト分布の覗き見）を避ける（leak-reviewer 指摘 D-1）。
    """
    if not monthly_aggs:
        raise ValueError("No monthly aggregations available to select zones")

    combined = pd.concat(monthly_aggs, ignore_index=True)
    train_only = combined[combined["date"] <= train_end]
    if train_only.empty:
        raise ValueError(
            f"No demand data on or before train_end={train_end.date()} "
            "for zone selection"
        )

    zone_totals = (
        train_only.groupby("zone", observed=True)["demand"]
        .sum()
        .sort_values(ascending=False)
    )

    # Manhattan ゾーン内のみ対象（念のため絞る）
    zone_totals = zone_totals[zone_totals.index.isin(manhattan_zones)]

    top_zones = zone_totals.head(top_n).index.tolist()

    logger.info(
        "=== Zone selection (top %d by TRAIN-period demand, date <= %s) ===",
        top_n,
        train_end.date(),
    )
    for rank, (zone_id, total) in enumerate(zone_totals.head(top_n).items(), 1):
        logger.info("  Rank %2d: zone=%d  train_demand=%d", rank, zone_id, total)

    return top_zones


# ---------------------------------------------------------------------------
# メイン集計
# ---------------------------------------------------------------------------

def build_daily_demand(
    cfg: dict[str, Any], intermediate_path: Path | None = None
) -> pd.DataFrame:
    """
    config に従って TLC データを取得・集計し、完全グリッドの日次需要 DataFrame を返す。

    intermediate_path を渡すと、全 Manhattan ゾーン・全期間の日次需要を中間データ
    として保存する。これにより、split 変更などでゾーンを選び直したくなった際に
    TLC を再ダウンロードせず再選択できる。
    """
    period_cfg = cfg["period"]
    zones_cfg = cfg["zones"]

    start_date = pd.Timestamp(period_cfg["start_date"])
    end_date = pd.Timestamp(period_cfg["end_date"])
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    exclude_ids = zones_cfg.get("exclude_ids", [])
    top_n = zones_cfg["top_n"]

    # Step 1: Manhattan ゾーン取得
    manhattan_zones = fetch_manhattan_zones(exclude_ids)

    # Step 2: 月ごとに取得・集計
    months = pd.period_range(start=start_date, end=end_date, freq="M")
    monthly_aggs: list[pd.DataFrame] = []

    for period in months:
        agg = _fetch_month_demand(
            year=period.year,
            month=period.month,
            manhattan_zones=manhattan_zones,
            start_date=start_date,
            end_date=end_date,
        )
        if not agg.empty:
            monthly_aggs.append(agg)

    if not monthly_aggs:
        raise RuntimeError("No data could be fetched from TLC. Check network access.")

    # Step 3: トップ N ゾーン選択（train 期間のみで決定）
    top_zones = select_top_zones(monthly_aggs, manhattan_zones, top_n, train_end)

    # Step 4: 全 Manhattan ゾーンの集計を統合・再集約
    all_agg = pd.concat(monthly_aggs, ignore_index=True)
    del monthly_aggs  # メモリ解放

    # 月をまたいだデータが複数月の集計に含まれる場合（例: 12/31 が 1月ファイルに混入）
    # concat 後に同一 (date, zone) の重複が生じうるので groupby で再集約する
    all_agg = (
        all_agg.groupby(["date", "zone"], observed=True)["demand"]
        .sum()
        .reset_index()
    )

    # 全 Manhattan ゾーン・全期間の中間データを保存（将来の再選択で再DL不要に）
    if intermediate_path is not None:
        save_daily_demand(all_agg, intermediate_path)
        logger.info(
            "Saved all-Manhattan-zone intermediate (%d zones) to %s",
            all_agg["zone"].nunique(),
            intermediate_path,
        )

    # 選択ゾーンに絞る
    all_agg = all_agg[all_agg["zone"].isin(top_zones)]
    logger.info(
        "Combined aggregation: %d unique (date, zone) pairs for %d selected zones",
        len(all_agg),
        len(top_zones),
    )

    # Step 5: 完全グリッド作成
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(all_dates)
    n_zones = len(top_zones)

    grid = pd.MultiIndex.from_product(
        [all_dates, top_zones], names=["date", "zone"]
    ).to_frame(index=False)

    # demand を結合（欠損を 0 で埋める）
    grid = grid.merge(all_agg, on=["date", "zone"], how="left")
    grid["demand"] = grid["demand"].fillna(0)

    # 型を整える
    grid["date"] = pd.to_datetime(grid["date"])
    grid["zone"] = grid["zone"].astype(int)
    grid["demand"] = grid["demand"].astype(float)

    # Step 6: assert チェック
    assert len(grid) == n_days * n_zones, (
        f"Grid size mismatch: expected {n_days}x{n_zones}={n_days * n_zones}, "
        f"got {len(grid)}"
    )
    assert grid[["date", "zone"]].duplicated().sum() == 0, "Duplicate (date, zone) pairs found"
    assert grid["demand"].isna().sum() == 0, "NaN values found in demand column"
    assert (grid["demand"] >= 0).all(), "Negative demand values found"

    logger.info(
        "Grid assertion passed: %d days x %d zones = %d rows",
        n_days,
        n_zones,
        len(grid),
    )

    return grid[["date", "zone", "demand"]]


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------

def save_daily_demand(df: pd.DataFrame, path: str) -> None:
    """DataFrame を Parquet として保存する。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    logger.info("Saved daily_demand to %s  shape=%s", p, df.shape)


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = load_config()

    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / cfg["paths"]["raw_demand"]
    intermediate_path = raw_path.parent / "_manhattan_daily_all.parquet"

    df = build_daily_demand(cfg, intermediate_path=intermediate_path)
    save_daily_demand(df, raw_path)

    print("\n=== daily_demand summary ===")
    print(f"shape      : {df.shape}")
    print(f"columns    : {df.columns.tolist()}")
    print(f"date range : {df['date'].min()} ~ {df['date'].max()}")
    print(f"zones      : {sorted(df['zone'].unique().tolist())}")
    print(df.head(10).to_string(index=False))
