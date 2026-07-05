"""
ingest_subway.py — Phase B: MTA地下鉄（Socrata, dataset wujg-7c2s）から
システム全体（全ボロー・全支払い方法合算）の時間帯別利用者数を取得し、
Citi Bike (src/ingest_segments.py) と同じ日次セグメント形式に集計する。

- APIキー不要。SoQL でサーバー側集計（$select=sum(ridership), $group=transit_timestamp）
  し、生の乗降レコードを丸ごと落とさない。
- $offset を page_limit ずつ増やし、空配列が返るまでページングする（将来レコード数が
  増えても壊れない設計。1回で収まっていても必ずループにする）。
- HTTPリトライは nyc-taxi-demand/src/ingest.py の _fetch_month_demand と同じパターン
  （指数バックオフ 5*2**(attempt-1) 秒、既定4回、尽きたら RuntimeError）。
  取得失敗をサイレントに0埋めしない（このリポジトリ全体の規律）。
- weekday_am_hours/weekend_mid_hours は config.yaml トップレベルの
  segments.roundtrip から読む（Citi Bike側と時間窓定義を共有し、ズレを防ぐ）。

出力: data/phaseB_subway_segments.parquet  [date, segment, n]（long形式、n は int）
  - all: 全日の日次合計
  - time_wdAM: 平日(dow<5)かつ hour が weekday_am_hours に含まれる行の日次合計
  - time_weMID: 休日(dow>=5)かつ hour が weekend_mid_hours に含まれる行の日次合計
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# HTTP取得（リトライ付き）
# ---------------------------------------------------------------------------

def _fetch_page(url: str, params: dict, retries: int = 4) -> list[dict]:
    """
    1ページ分の Socrata レスポンス(JSON配列)を取得する。

    指数バックオフ付きでリトライし（CloudFront/Socrataの一時的なレート制限・
    接続リセット対策）、全リトライ失敗時は RuntimeError を送出する。
    ページ取得をサイレントにスキップすると、その分の日付がグリッド上で
    0埋めされバイアスが入るため、絶対にスキップしない。
    """
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            last_err = e
            wait = 5 * (2 ** (attempt - 1))  # 5, 10, 20, 40 秒
            logger.warning(
                "  attempt %d/%d failed for offset=%s: %s%s",
                attempt,
                retries,
                params.get("$offset"),
                e,
                f" (retrying in {wait}s)" if attempt < retries else "",
            )
            if attempt < retries:
                time.sleep(wait)

    raise RuntimeError(
        f"MTA subway fetch failed for offset={params.get('$offset')} after {retries} "
        f"attempts: {last_err}. Not skipping silently (would zero-fill the period and "
        "bias the data)."
    )


def fetch_subway_hourly(period: dict, subway_cfg: dict) -> pd.DataFrame:
    """
    Socrata SoQL でシステム全体・時間帯別合計利用者数をページングしながら取得する。
    Returns: DataFrame [transit_timestamp (datetime64), ridership (float)]
    """
    url = subway_cfg["socrata_url"]
    limit = subway_cfg["page_limit"]
    where = (
        f"transit_timestamp between '{period['start_date']}T00:00:00' "
        f"and '{period['end_date']}T23:59:59'"
    )

    pages: list[pd.DataFrame] = []
    offset = 0
    while True:
        params = {
            "$select": "transit_timestamp, sum(ridership) as ridership",
            "$where": where,
            "$group": "transit_timestamp",
            "$order": "transit_timestamp",
            "$limit": limit,
            "$offset": offset,
        }
        logger.info("Fetching MTA subway hourly ridership (offset=%d)", offset)
        data = _fetch_page(url, params)
        if not data:
            break
        pages.append(pd.DataFrame(data))
        offset += limit

    if not pages:
        raise RuntimeError(
            "MTA subway fetch returned no data at all for the requested period "
            f"{period['start_date']} ~ {period['end_date']}"
        )

    df = pd.concat(pages, ignore_index=True)
    df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])
    df["ridership"] = df["ridership"].astype(float)
    logger.info(
        "subway hourly: %d rows fetched (%s ~ %s)",
        len(df), df["transit_timestamp"].min(), df["transit_timestamp"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# セグメント集計（Citi Bike ingest_segments.py の軸2と同一ロジック）
# ---------------------------------------------------------------------------

def _segments_from_hourly(df: pd.DataFrame, am_hours: list[int], mid_hours: list[int]) -> pd.DataFrame:
    d = df.assign(
        date=df["transit_timestamp"].dt.normalize(),
        hour=df["transit_timestamp"].dt.hour,
        dow=df["transit_timestamp"].dt.dayofweek,
    )
    weekend = d["dow"] >= 5

    parts = {
        "all": d,
        "time_wdAM": d[(~weekend) & d["hour"].isin(am_hours)],
        "time_weMID": d[weekend & d["hour"].isin(mid_hours)],
    }

    rows = []
    for seg, sub in parts.items():
        s = sub.groupby("date")["ridership"].sum()
        for dt, n in s.items():
            rows.append({"date": dt, "segment": seg, "n": int(round(n))})
    out = pd.DataFrame(rows).sort_values(["segment", "date"]).reset_index(drop=True)
    return out


def ingest(cfg: dict) -> pd.DataFrame:
    rt_cfg = cfg["segments"]["roundtrip"]
    am_hours = rt_cfg["weekday_am_hours"]
    mid_hours = rt_cfg["weekend_mid_hours"]

    hourly = fetch_subway_hourly(cfg["phase_b"]["period"], cfg["phase_b"]["subway"])
    out = _segments_from_hourly(hourly, am_hours, mid_hours)
    logger.info(
        "segments: %s",
        out.groupby("segment")["n"].agg(["count", "mean"]).round(0).to_dict(),
    )
    return out


def main() -> None:
    cfg = load_config()
    out = ingest(cfg)
    dest = PROJECT_ROOT / cfg["phase_b"]["paths"]["subway_segments"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    logger.info("Saved %s shape=%s", dest, out.shape)
    print(out.groupby("segment").agg(days=("n", "size"), mean_per_day=("n", "mean")).round(0).to_string())


if __name__ == "__main__":
    main()
