"""
events.py - NYC Open Data から許可イベント情報を取得し日次集計する

エンドポイント: NYC Permitted Event Information (bkfu-528j)
  https://data.cityofnewyork.us/resource/bkfu-528j.json

  このデータセットは 2022〜2024 年の期間をカバーする。
  Socrata の date_trunc_ymd() + GROUP BY を月単位で発行し、
  各リクエストのタイムアウトを回避しながら日次集計を取得する。

日次の event_count（許可イベント件数）を返す。
各月はバックオフ付きでリトライし、取得失敗月が 1 つでも残れば RuntimeError で
停止する（サイレント 0 埋めは系統的バイアスの原因になるため行わない）。全月の
取得に成功した場合のみ、グリッド上に残る欠損日を「真の 0 件」として 0 で埋める。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# NYC Permitted Event Information (historical, 2008〜現在)
NYC_EVENTS_URL = "https://data.cityofnewyork.us/resource/bkfu-528j.json"

def _fetch_month_events(
    year: int, month: int, timeout: int = 60, retries: int = 4
) -> list[dict]:
    """
    指定月の日次集計（event_day, cnt）を Socrata API から取得する。

    一時的なタイムアウト/HTTP エラーはバックオフ付きでリトライする。
    また「HTTP 200 だが 0 件」は異常として扱う（NYC permitted events は
    毎月必ず存在するため、空応答 = 部分障害とみなす）。全リトライ失敗時は
    RuntimeError を送出し、呼び出し元が「取得失敗月」として扱えるようにする。
    こうすることで、取得失敗を「真のイベント 0 件」と混同してサイレントに
    0 埋めする（系統的バイアスの原因）ことを防ぐ。
    """
    month_start = f"{year}-{month:02d}-01"
    # 月末を計算
    if month == 12:
        next_month = pd.Timestamp(f"{year + 1}-01-01")
    else:
        next_month = pd.Timestamp(f"{year}-{month + 1:02d}-01")
    month_end = (next_month - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    params = {
        # COUNT(*) は 1 イベントが複数行（時間帯・区画・precinct 等）に展開され
        # 約 20〜27 倍に水増しされるため、event_id で一意化して「その日に開始する
        # 個別イベント数」を数える（leak-reviewer の二次指摘への対応）。
        "$select": (
            "date_trunc_ymd(start_date_time) AS event_day, "
            "COUNT(DISTINCT event_id) AS cnt"
        ),
        "$where": (
            f"start_date_time >= '{month_start}T00:00:00' "
            f"AND start_date_time <= '{month_end}T23:59:59'"
        ),
        "$group": "event_day",
        "$order": "event_day ASC",
        "$limit": 100,  # 月あたり最大 31 日
    }

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(NYC_EVENTS_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            records = resp.json()
            if not records:
                # 200 だが 0 件 = 部分障害とみなしリトライ
                raise RuntimeError("empty result (0 daily records) for a full month")
            return records
        except (requests.RequestException, ValueError, RuntimeError) as e:
            last_err = e
            wait = 2 ** attempt  # 2, 4, 8, 16 秒
            logger.warning(
                "  attempt %d/%d failed for %d-%02d: %s%s",
                attempt,
                retries,
                year,
                month,
                e,
                f" (retrying in {wait}s)" if attempt < retries else "",
            )
            if attempt < retries:
                time.sleep(wait)

    raise RuntimeError(
        f"events fetch failed for {year}-{month:02d} "
        f"after {retries} attempts: {last_err}"
    )


def fetch_events(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    NYC Open Data (bkfu-528j) から許可イベント情報を取得し、日次集計を返す。

    月単位で Socrata GROUP BY クエリを発行し、各月をリトライ付きで取得する。
    取得に失敗した月が 1 つでもあれば RuntimeError を送出して run を止める。
    これは、取得失敗を「真のイベント 0 件」と取り違えてサイレントに 0 埋めし、
    train/val/test を横断する系統的バイアスを混入させることを防ぐため
    （leak-reviewer 指摘 R-1）。全月の取得に成功した場合のみ、グリッド上に
    残る欠損日を「真の 0 件」として 0 で埋める。

    Returns
    -------
    DataFrame with:
      - date (datetime64[ns])
      - event_count (int)  : その日の許可イベント件数
    """
    period_cfg = cfg["period"]
    start_date = period_cfg["start_date"]
    end_date = period_cfg["end_date"]

    logger.info(
        "Fetching NYC permitted events from %s (%s ~ %s)",
        NYC_EVENTS_URL,
        start_date,
        end_date,
    )

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    months = pd.period_range(start=start_date, end=end_date, freq="M")

    all_records: list[dict] = []
    failed_months: list[str] = []

    for period in months:
        try:
            records = _fetch_month_events(period.year, period.month)
            logger.info(
                "  %d-%02d: %d daily records", period.year, period.month, len(records)
            )
            all_records.extend(records)
        except RuntimeError as e:
            logger.error(
                "Failed to fetch events for %d-%02d: %s",
                period.year,
                period.month,
                e,
            )
            failed_months.append(f"{period.year}-{period.month:02d}")

    # 取得失敗月があれば fail させる（サイレント 0 埋め禁止）
    if failed_months:
        raise RuntimeError(
            f"Events fetch failed for {len(failed_months)} month(s): {failed_months}. "
            "Re-run when the NYC Open Data API is reachable. These months must NOT be "
            "silently filled with 0 — doing so injects a systematic bias across "
            "train/val/test (leak-reviewer finding R-1)."
        )

    events_df = pd.DataFrame(all_records)
    logger.info("Total daily aggregated records from API: %d rows", len(events_df))

    events_df["date"] = pd.to_datetime(
        events_df["event_day"], errors="coerce"
    ).dt.normalize()
    events_df["event_count"] = (
        pd.to_numeric(events_df["cnt"], errors="coerce").fillna(0).astype(int)
    )
    events_df = events_df[["date", "event_count"]].dropna(subset=["date"])

    # 期間内に絞る
    events_df = events_df[
        (events_df["date"] >= start_ts) & (events_df["date"] <= end_ts)
    ]

    # 全日付グリッドに合わせる。全月の取得に成功しているので、ここで残る欠損は
    # 「その日にイベント情報が無い」= 真の 0 件であり、0 埋めは正当。
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    date_grid = pd.DataFrame({"date": all_dates})
    daily_events = date_grid.merge(events_df, on="date", how="left")
    daily_events["event_count"] = daily_events["event_count"].fillna(0).astype(int)

    logger.info(
        "Events processed: %d event-days with count>0, max=%d, total_grid=%d days",
        (daily_events["event_count"] > 0).sum(),
        int(daily_events["event_count"].max()),
        len(daily_events),
    )

    return daily_events[["date", "event_count"]]


# ============================================================================
# ゾーン解像イベント（条件 E 用）
# ----------------------------------------------------------------------------
# 上の fetch_events() は「全市スカラー」の event_count（条件 C/D 用）であり一切
# 変更しない。ここでは Manhattan 限定でイベントを取得し、位置情報 police_precinct
# を precinct→taxi zone の面積重みクロスウォークでゾーンに割り付けて、(date, zone)
# 別の event_count を作る（条件 E 用）。これにより外部情報に空間解像度を与える。
# ============================================================================

# 本体パイプラインは geopandas を持たない。ゾーン割付はビルド済み CSV
# （scripts/build_precinct_zone_crosswalk.py が生成）だけに依存する。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_precinct_zone_crosswalk(path: str | Path) -> pd.DataFrame:
    """precinct→LocationID の面積重みクロスウォーク CSV を読む（cwd に依らず解決）。"""
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"precinct→zone crosswalk not found: {p}. "
            "Run scripts/build_precinct_zone_crosswalk.py first."
        )
    cw = pd.read_csv(p)
    cw["precinct"] = cw["precinct"].astype(int)
    cw["LocationID"] = cw["LocationID"].astype(int)
    return cw[["precinct", "LocationID", "weight"]]


def _fetch_month_events_zone(
    year: int, month: int, timeout: int = 60, retries: int = 4
) -> list[dict]:
    """
    指定月の Manhattan イベントを (event_id, event_day, police_precinct) 単位で取得。

    1 イベントは複数行（時間帯・区画）に展開されるため $group で event_id 単位に畳む。
    fetch_events() と同じく、空応答は部分障害とみなしリトライし、全失敗で RuntimeError。
    """
    month_start = f"{year}-{month:02d}-01"
    if month == 12:
        next_month = pd.Timestamp(f"{year + 1}-01-01")
    else:
        next_month = pd.Timestamp(f"{year}-{month + 1:02d}-01")
    month_end = (next_month - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    params = {
        "$select": (
            "event_id, "
            "date_trunc_ymd(start_date_time) AS event_day, "
            "police_precinct"
        ),
        "$where": (
            f"start_date_time >= '{month_start}T00:00:00' "
            f"AND start_date_time <= '{month_end}T23:59:59' "
            "AND event_borough = 'Manhattan' "
            "AND police_precinct IS NOT NULL"
        ),
        "$group": "event_id, event_day, police_precinct",
        "$order": "event_day ASC",
        "$limit": 50000,
    }

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(NYC_EVENTS_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            records = resp.json()
            if not records:
                raise RuntimeError("empty result (0 records) for a full month")
            return records
        except (requests.RequestException, ValueError, RuntimeError) as e:
            last_err = e
            wait = 2 ** attempt
            logger.warning(
                "  [zone] attempt %d/%d failed for %d-%02d: %s%s",
                attempt,
                retries,
                year,
                month,
                e,
                f" (retrying in {wait}s)" if attempt < retries else "",
            )
            if attempt < retries:
                time.sleep(wait)

    raise RuntimeError(
        f"zone events fetch failed for {year}-{month:02d} "
        f"after {retries} attempts: {last_err}"
    )


def _parse_precincts(raw: str) -> list[int]:
    """police_precinct 文字列（例 "17," / "17,18,"）を precinct 番号リストに分解する。"""
    out: list[int] = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
    return out


def fetch_events_by_zone(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Manhattan のイベントを取得し、police_precinct を面積重みクロスウォークで taxi zone
    に割り付けて (date, zone) 別の event_count を返す（条件 E 用）。

    数え方:
      - 複数 precinct にまたがるイベントは各 precinct で全数カウントする（合意した既定）。
        zone への寄与 = sum_over_precincts_of_event( weight(precinct -> zone) )。
      - precinct→zone は面積按分のため、単一 precinct のイベントでも複数ゾーンへ少数に
        分配され、event_count は連続値（ゾーン×日あたりの期待イベント強度）になる。

    リーク/バイアス防止:
      - 取得失敗月が 1 つでもあれば RuntimeError（サイレント 0 埋め禁止, R-1）。
      - イベントは開始日(start_date_time)基準で計上（fetch_events と同じ既知の限界）。

    Returns
    -------
    DataFrame[date, zone, event_count]  （event_count>0 の行のみ。欠損 (date,zone) は
      真の 0 件で、merge 側の left join + fillna(0) が埋める。）
    """
    crosswalk = load_precinct_zone_crosswalk(cfg["paths"]["precinct_zone_crosswalk"])

    period_cfg = cfg["period"]
    start_date = period_cfg["start_date"]
    end_date = period_cfg["end_date"]
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    logger.info(
        "Fetching ZONE-resolved Manhattan events (%s ~ %s) via precinct crosswalk",
        start_date,
        end_date,
    )

    months = pd.period_range(start=start_date, end=end_date, freq="M")
    all_records: list[dict] = []
    failed_months: list[str] = []

    for period in months:
        try:
            records = _fetch_month_events_zone(period.year, period.month)
            logger.info(
                "  [zone] %d-%02d: %d (event,precinct) rows",
                period.year,
                period.month,
                len(records),
            )
            all_records.extend(records)
        except RuntimeError as e:
            logger.error(
                "Failed to fetch zone events for %d-%02d: %s",
                period.year,
                period.month,
                e,
            )
            failed_months.append(f"{period.year}-{period.month:02d}")

    if failed_months:
        raise RuntimeError(
            f"Zone events fetch failed for {len(failed_months)} month(s): {failed_months}. "
            "Re-run when the NYC Open Data API is reachable. These months must NOT be "
            "silently filled with 0 (systematic bias across train/val/test, R-1)."
        )

    raw_df = pd.DataFrame(all_records)
    raw_df["date"] = pd.to_datetime(raw_df["event_day"], errors="coerce").dt.normalize()
    raw_df = raw_df.dropna(subset=["date"])
    raw_df = raw_df[(raw_df["date"] >= start_ts) & (raw_df["date"] <= end_ts)]

    # police_precinct を分解 → (event_id, date, precinct) に展開。
    raw_df["precincts"] = raw_df["police_precinct"].map(_parse_precincts)
    exploded = raw_df.explode("precincts").rename(columns={"precincts": "precinct"})
    exploded = exploded.dropna(subset=["precinct"])
    exploded["precinct"] = exploded["precinct"].astype(int)

    # 全数カウント: 1 イベントは触れる各 precinct で 1 回（重複行は畳む）。
    pairs = exploded[["event_id", "date", "precinct"]].drop_duplicates()

    # precinct→zone 面積重みで寄与を割り付け、(date, zone) で合算。
    contrib = pairs.merge(crosswalk, on="precinct", how="inner")
    contrib["event_count"] = contrib["weight"]
    zone_events = (
        contrib.groupby(["date", "LocationID"])["event_count"]
        .sum()
        .reset_index()
        .rename(columns={"LocationID": "zone"})
    )
    zone_events = zone_events[zone_events["event_count"] > 0].reset_index(drop=True)

    logger.info(
        "Zone events: %d distinct events -> %d (date,zone) rows, "
        "zones touched=%d, max=%.2f",
        pairs["event_id"].nunique(),
        len(zone_events),
        zone_events["zone"].nunique(),
        zone_events["event_count"].max(),
    )

    return zone_events[["date", "zone", "event_count"]]


if __name__ == "__main__":
    cfg = load_config()
    df = fetch_events(cfg)
    print("\n=== events summary (city-wide scalar, conditions C/D) ===")
    print(f"shape        : {df.shape}")
    print(f"columns      : {df.columns.tolist()}")
    print(f"event_count>0: {(df['event_count'] > 0).sum()} days")
    print(df[df["event_count"] > 0].head(10).to_string(index=False))

    zdf = fetch_events_by_zone(cfg)
    print("\n=== zone-resolved events summary (condition E) ===")
    print(f"shape        : {zdf.shape}")
    print(f"columns      : {zdf.columns.tolist()}")
    print(f"(date,zone)>0: {len(zdf)} rows over {zdf['zone'].nunique()} zones")
    print(zdf.sort_values('event_count', ascending=False).head(10).to_string(index=False))
