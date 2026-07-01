"""
build_precinct_zone_crosswalk.py — NYPD precinct → TLC taxi zone の面積重みクロスウォークを生成する。

【位置づけ（重要）】
本スクリプトは「一度だけ」実行して config/precinct_zone_crosswalk.csv を作るための
ビルドツールであり、本体パイプライン（ingest/merge/features/train）からは import しない。
そのため geopandas/shapely/pyproj への依存はここに閉じ込めてある。生成された CSV だけを
本体が読む。これにより「ゾーン解像イベント」を、ジオコーディング無し・再現可能な根拠つきで
events.py から利用できる。

【何を作るか】
許可イベントは位置情報として police_precinct（例 "17,"）しか持たず緯度経度が無い。そこで
「precinct P に属するイベントを taxi zone Z にどう割り付けるか」を面積按分で定める:

    weight(P -> Z) = area(P ∩ Z) / area(P)

各 precinct について全 zone にわたる weight の和は（水域・スライバを除き）ほぼ 1 に
正規化される（＝単一 precinct 内での二重計上を防ぐ）。複数 precinct にまたがる
イベントは events.py 側で precinct ごとに分解し、本表の weight を各 precinct で
全数合算する（合意した既定。平均はしない）。したがって K 個の precinct にまたがる
1 イベントは全ゾーン合計で質量 K になる（イベント単位の質量保存ではない点に注意）。

【入力】
  data/geo/taxi_zones/taxi_zones/taxi_zones.shp   (TLC 公式, EPSG:2263, LocationID/borough)
  data/geo/police_precincts.geojson               (NYC Open Data y76i-bdw7, precinct)

【出力】
  config/precinct_zone_crosswalk.csv  (precinct:int, LocationID:int, weight:float)

実行:
  .venv/Scripts/python.exe scripts/build_precinct_zone_crosswalk.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAXI_ZONES_SHP = PROJECT_ROOT / "data/geo/taxi_zones/taxi_zones/taxi_zones.shp"
PRECINCTS_GEOJSON = PROJECT_ROOT / "data/geo/police_precincts.geojson"
OUT_CSV = PROJECT_ROOT / "config/precinct_zone_crosswalk.csv"

# 面積計算は等積に近い投影座標系で行う。TLC taxi_zones は EPSG:2263
# (NAD83 / New York Long Island, 単位フィート)。precinct を同 CRS に揃える。
WORKING_CRS = "EPSG:2263"

# 面積按分後、ごく僅かな overlap（スライバ）を捨てる閾値。precinct 面積に対する比率。
MIN_WEIGHT = 1e-4


def build_crosswalk() -> pd.DataFrame:
    tz = gpd.read_file(TAXI_ZONES_SHP)[["LocationID", "borough", "geometry"]]
    pre = gpd.read_file(PRECINCTS_GEOJSON)[["precinct", "geometry"]]
    logger.info("taxi zones=%d, precincts=%d", len(tz), len(pre))

    # CRS を投影座標系に統一（precinct geojson は EPSG:4326）。
    tz = tz.to_crs(WORKING_CRS)
    pre = pre.to_crs(WORKING_CRS)

    # 無効ジオメトリ（自己交差等）を buffer(0) で修復し、overlay の失敗を防ぐ。
    tz["geometry"] = tz.geometry.buffer(0)
    pre["geometry"] = pre.geometry.buffer(0)

    # precinct 番号を int 化（events 側の "17" と突き合わせるため）。
    pre["precinct"] = pd.to_numeric(pre["precinct"], errors="coerce").astype("Int64")
    pre = pre.dropna(subset=["precinct"])
    pre["precinct"] = pre["precinct"].astype(int)

    # precinct 総面積（分母）。
    pre["precinct_area"] = pre.geometry.area

    # precinct × zone の交差片を作り、各片の面積を測る。
    inter = gpd.overlay(
        pre[["precinct", "precinct_area", "geometry"]],
        tz[["LocationID", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    inter["inter_area"] = inter.geometry.area
    inter["weight"] = inter["inter_area"] / inter["precinct_area"]

    # スライバ除去 → precinct ごとに weight を再正規化（和を 1 に）。
    inter = inter[inter["weight"] >= MIN_WEIGHT].copy()
    inter["weight"] = inter["weight"] / inter.groupby("precinct")["weight"].transform("sum")

    crosswalk = (
        inter[["precinct", "LocationID", "weight"]]
        .sort_values(["precinct", "weight"], ascending=[True, False])
        .reset_index(drop=True)
    )
    crosswalk["LocationID"] = crosswalk["LocationID"].astype(int)
    return crosswalk


def main() -> None:
    crosswalk = build_crosswalk()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(OUT_CSV, index=False)

    n_prec = crosswalk["precinct"].nunique()
    n_zone = crosswalk["LocationID"].nunique()
    wsum = crosswalk.groupby("precinct")["weight"].sum()
    logger.info(
        "Wrote %s: %d rows, %d precincts -> %d zones",
        OUT_CSV, len(crosswalk), n_prec, n_zone,
    )
    logger.info(
        "weight-sum per precinct: min=%.4f max=%.4f (should be ~1.0)",
        wsum.min(), wsum.max(),
    )
    # 確認用に Manhattan を多く含む低番号 precinct の割付を表示。
    sample = crosswalk[crosswalk["precinct"].isin([1, 14, 17, 18])]
    print("\n=== sample precinct -> zone weights ===")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
