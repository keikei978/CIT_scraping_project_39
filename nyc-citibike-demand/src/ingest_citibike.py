"""
ingest_citibike.py — Phase 1: Citi Bike トリップを日次×システム合算の需要に集計する。

設計:
  - Citi Bike 公式 S3 の NYC 年次 zip（config.citibike.files）を取得。
  - 1ファイルが数千万行になり得るため、zip 内の各 CSV を **chunk 読み**して
    started_at→date, member_casual で集計し、生トリップは保持しない（メモリ安全）。
  - 出力: data/daily_demand.parquet  [date, member, casual, total]
  - member/casual を最初から保持（member=通勤=ラグ駆動 / casual=レク=天候駆動 の対比用）。

リーク/バイアス防止（タクシー版 R-1 の教訓）:
  - 取得・解凍・解析に失敗したら RuntimeError で停止（欠損をサイレント 0 埋めしない）。
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNK = 500_000
USECOLS = ["started_at", "member_casual"]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _download(url: str, dest: Path, retries: int = 8) -> Path:
    """
    巨大 zip を Range レジューム付きでダウンロード（接続リセットに強い）。
    - サーバの Content-Length と一致したら完了とみなしスキップ（部分ファイルは続きから）。
    - ChunkedEncodingError / ConnectionError はバックオフ付きで再試行し、
      その都度ローカルの続きバイトから再開する。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = int(requests.head(url, timeout=60).headers.get("Content-Length", 0))

    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if total and have >= total:
            logger.info("  complete %s (%.0f MB)", dest.name, have / 1e6)
            return dest
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=600, headers=headers) as r:
                r.raise_for_status()
                mode = "ab" if have else "wb"
                with open(dest, mode) as f:
                    for block in r.iter_content(chunk_size=1 << 20):
                        f.write(block)
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError) as e:
            wait = min(2 ** attempt, 30)
            logger.warning("  download interrupted (%s) attempt %d/%d, resume in %ds",
                           type(e).__name__, attempt, retries, wait)
            time.sleep(wait)
            continue
    final = dest.stat().st_size if dest.exists() else 0
    if not total or final < total:
        raise RuntimeError(f"download incomplete for {dest.name}: {final}/{total} bytes")
    logger.info("  saved %s (%.0f MB)", dest.name, final / 1e6)
    return dest


def _iter_csv_members(zf: zipfile.ZipFile):
    """zip 内の CSV を列挙（ネスト zip や __MACOSX を吸収）。"""
    for name in zf.namelist():
        if "__MACOSX" in name or name.endswith("/"):
            continue
        if name.lower().endswith(".csv"):
            yield ("csv", name)
        elif name.lower().endswith(".zip"):
            yield ("zip", name)


def _aggregate_csv(fobj) -> pd.DataFrame:
    """1 CSV を chunk 読みして date×member_casual の件数に集計。"""
    parts = []
    for chunk in pd.read_csv(fobj, usecols=USECOLS, chunksize=CHUNK, low_memory=False):
        chunk["date"] = pd.to_datetime(chunk["started_at"], errors="coerce").dt.normalize()
        chunk = chunk.dropna(subset=["date"])
        g = chunk.groupby(["date", "member_casual"]).size()
        parts.append(g)
    if not parts:
        raise RuntimeError("CSV yielded no rows")
    return pd.concat(parts).groupby(level=[0, 1]).sum().rename("n").reset_index()


def _aggregate_zip(zip_path: Path) -> pd.DataFrame:
    """年次 zip 全体を集計して date×member_casual 件数を返す。"""
    accum = []
    with zipfile.ZipFile(zip_path) as zf:
        members = list(_iter_csv_members(zf))
        if not members:
            raise RuntimeError(f"no CSV found inside {zip_path.name}")
        for kind, name in members:
            if kind == "csv":
                with zf.open(name) as fobj:
                    accum.append(_aggregate_csv(fobj))
            else:  # nested zip
                with zf.open(name) as nested:
                    with zipfile.ZipFile(io.BytesIO(nested.read())) as nzf:
                        inner = [n for n in nzf.namelist()
                                 if n.lower().endswith(".csv") and "__MACOSX" not in n]
                        for icsv in inner:
                            with nzf.open(icsv) as fobj:
                                accum.append(_aggregate_csv(fobj))
            logger.info("    aggregated %s", name)
    return pd.concat(accum).groupby(["date", "member_casual"])["n"].sum().reset_index()


def ingest(cfg: dict) -> pd.DataFrame:
    cb = cfg["citibike"]
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    start = pd.Timestamp(cfg["period"]["start_date"])
    end = pd.Timestamp(cfg["period"]["end_date"])

    daily_parts = []
    for fname in cb["files"]:
        url = cb["s3_base"] + fname
        # ファイル別の集計結果をキャッシュ（再実行時に再集計を回避）。
        agg_path = raw_dir / f"_agg_{fname}.parquet"
        if agg_path.exists():
            logger.info("  reuse aggregated cache %s", agg_path.name)
            daily_parts.append(pd.read_parquet(agg_path))
            continue
        zpath = _download(url, raw_dir / fname)
        logger.info("  aggregating %s ...", fname)
        agg = _aggregate_zip(zpath)
        agg.to_parquet(agg_path, index=False)
        daily_parts.append(agg)

    long = pd.concat(daily_parts).groupby(["date", "member_casual"])["n"].sum().reset_index()
    wide = long.pivot(index="date", columns="member_casual", values="n").fillna(0)
    for col in ("member", "casual"):
        if col not in wide.columns:
            wide[col] = 0
    wide["total"] = wide[["member", "casual"]].sum(axis=1)
    wide = wide[(wide.index >= start) & (wide.index <= end)].sort_index()

    # 完全な日次グリッドを期待（欠損日があれば異常として報告＝サイレント 0 埋めしない）。
    full = pd.date_range(start, end, freq="D")
    missing = full.difference(wide.index)
    if len(missing) > 0:
        raise RuntimeError(
            f"{len(missing)} day(s) missing in Citi Bike daily series "
            f"(e.g. {list(missing[:5])}). Do NOT silently fill; investigate source."
        )

    out = wide.reset_index()[["date", "member", "casual", "total"]]
    logger.info(
        "Citi Bike daily: %d days, mean total=%d, member share=%.3f",
        len(out), int(out["total"].mean()), out["member"].sum() / out["total"].sum(),
    )
    return out


def main() -> None:
    cfg = load_config()
    out = ingest(cfg)
    dest = PROJECT_ROOT / cfg["paths"]["demand"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    logger.info("Saved %s  shape=%s", dest, out.shape)
    print(out.head().to_string(index=False))
    print("...")
    print(out.tail().to_string(index=False))


if __name__ == "__main__":
    main()
