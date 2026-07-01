"""
ingest_segments.py — Phase A: キャッシュ済み Citi Bike 生トリップを「セグメント別日次件数」に再集計。

裁量性プロキシ（同一モード=自転車内なので曝露・代替の向きは固定）を3軸で作る:
  軸1 ラウンドトリップ : rt_round(始点=終点駅, レジャーループ) vs rt_point(point-to-point, 通勤)
  軸2 時間帯×曜日      : time_wdAM(平日朝ラッシュ, 通勤) vs time_weMID(休日昼, レジャー)
  軸3 会員種別          : user_member(通勤寄り) vs user_casual(レジャー寄り)
  参照                 : all(全トリップ)

出力: data/segments_daily.parquet  [date, segment, n]（long 形式）
  - rt_*, user_*, all は全日。time_wdAM は平日のみ、time_weMID は休日のみ（構造的に）。
  - 再DLしない（既存 nyc-citibike-demand のキャッシュ zip を読むだけ）。
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNK = 500_000
USECOLS = ["started_at", "member_casual", "start_station_id", "end_station_id"]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _process_zip(zf: zipfile.ZipFile, am_hours, mid_hours, acc: dict) -> None:
    """zip 内の CSV を集計。ネストした zip（年次zip→月次zip→CSV）は再帰展開する。"""
    for name in zf.namelist():
        if "__MACOSX" in name or name.endswith("/"):
            continue
        low = name.lower()
        if low.endswith(".csv"):
            with zf.open(name) as fobj:
                _accumulate(fobj, am_hours, mid_hours, acc)
            logger.info("  done %s", name)
        elif low.endswith(".zip"):
            with zf.open(name) as nested:
                with zipfile.ZipFile(io.BytesIO(nested.read())) as nzf:
                    _process_zip(nzf, am_hours, mid_hours, acc)


def _accumulate(fobj, am_hours, mid_hours, acc: dict):
    """1 CSV を chunk 読みし、セグメント別 date 件数を acc(dict[str,Series]) に加算。"""
    for chunk in pd.read_csv(fobj, usecols=USECOLS, chunksize=CHUNK, low_memory=False):
        dt = pd.to_datetime(chunk["started_at"], errors="coerce")
        chunk = chunk.assign(date=dt.dt.normalize(), hour=dt.dt.hour, dow=dt.dt.dayofweek)
        chunk = chunk.dropna(subset=["date"])
        weekend = chunk["dow"] >= 5
        rt = (chunk["start_station_id"].astype("string")
              == chunk["end_station_id"].astype("string")).fillna(False)

        parts = {
            "all": chunk,
            "rt_round": chunk[rt],
            "rt_point": chunk[~rt],
            "user_member": chunk[chunk["member_casual"] == "member"],
            "user_casual": chunk[chunk["member_casual"] == "casual"],
            "time_wdAM": chunk[(~weekend) & chunk["hour"].isin(am_hours)],
            "time_weMID": chunk[weekend & chunk["hour"].isin(mid_hours)],
        }
        for seg, sub in parts.items():
            s = sub.groupby("date").size()
            acc[seg] = acc.get(seg, pd.Series(dtype="int64")).add(s, fill_value=0)


def ingest(cfg: dict) -> pd.DataFrame:
    raw_dir = (PROJECT_ROOT / cfg["citibike"]["raw_dir"]).resolve()
    rt_cfg = cfg["segments"]["roundtrip"]
    am_hours = rt_cfg["weekday_am_hours"]
    mid_hours = rt_cfg["weekend_mid_hours"]
    start = pd.Timestamp(cfg["period"]["start_date"])
    end = pd.Timestamp(cfg["period"]["end_date"])

    acc: dict[str, pd.Series] = {}
    for fname in cfg["citibike"]["files"]:
        zpath = raw_dir / fname
        if not zpath.exists():
            raise FileNotFoundError(f"cached zip not found: {zpath}")
        logger.info("aggregating %s ...", fname)
        with zipfile.ZipFile(zpath) as zf:
            _process_zip(zf, am_hours, mid_hours, acc)

    rows = []
    for seg, s in acc.items():
        s = s[(s.index >= start) & (s.index <= end)]
        for d, n in s.items():
            rows.append({"date": d, "segment": seg, "n": int(round(n))})
    out = pd.DataFrame(rows).sort_values(["segment", "date"]).reset_index(drop=True)
    logger.info("segments: %s", out.groupby("segment")["n"].agg(["count", "mean"]).round(0).to_dict())
    return out


def main() -> None:
    cfg = load_config()
    out = ingest(cfg)
    dest = PROJECT_ROOT / cfg["paths"]["segments"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    logger.info("Saved %s shape=%s", dest, out.shape)
    print(out.groupby("segment").agg(days=("n", "size"), mean_per_day=("n", "mean")).round(0).to_string())


if __name__ == "__main__":
    main()
