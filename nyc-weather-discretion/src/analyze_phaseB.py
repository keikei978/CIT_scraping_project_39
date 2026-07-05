"""
analyze_phaseB.py — Phase B: 自転車(曝露あり) x 地下鉄(曝露なし) の2x2で、
「曝露/モード代替」と「裁量性（通勤 vs レジャー）」を切り分ける。

行(mode) = bike vs subway: 同じsegment内でこの行が大きく動くなら、曝露・モード代替の影響。
列(segment) = commute(time_wdAM) vs leisure(time_weMID): 両modeで同じ方向
（レジャー側がより雨に弱い）に動くなら、裁量性が主因であることの支持材料。

analyze_segment() は Phase A (src/analyze_segments.py) のものをそのまま再利用する
（無改造・再実装しない）。Phase B 用の config は adapter dict で形を合わせるだけ。

事前にどちらが正しいと決めつけず、数値をそのまま print する
（本プロジェクト全体の「事前登録」精神を踏襲）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from src.analyze_segments import analyze_segment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Phase A 側と同じ時間帯セグメント定義（軸2: 時間帯x曜日）を bike/subway 共通ラベルにマップ。
SEGMENT_LABELS = {"time_wdAM": "commute", "time_weMID": "leisure"}


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    # analyze_segment() は cfg["split"]/cfg["weather"]["daily_vars"]/cfg["baseline"]["harmonics"]
    # を読む設計。Phase B 用に split だけ差し替えた adapter dict を渡す（関数は無改造）。
    analyze_cfg = {
        "split": cfg["phase_b"]["split"],
        "weather": cfg["weather"],
        "baseline": cfg["baseline"],
    }

    merged = pd.read_parquet(PROJECT_ROOT / cfg["phase_b"]["paths"]["merged"])
    merged["date"] = pd.to_datetime(merged["date"])

    # phase_b.split の日付が実データ範囲に収まっているか確認（範囲外なら即エラーで気付けるように）。
    train_end = pd.Timestamp(analyze_cfg["split"]["train_end"])
    val_end = pd.Timestamp(analyze_cfg["split"]["val_end"])
    dmin, dmax = merged["date"].min(), merged["date"].max()
    assert dmin <= train_end < val_end <= dmax, (
        f"phase_b.split の日付がデータ範囲外です: data=[{dmin.date()}, {dmax.date()}], "
        f"train_end={train_end.date()}, val_end={val_end.date()}"
    )
    logger.info("data range=[%s, %s]  train_end=%s  val_end=%s",
                dmin.date(), dmax.date(), train_end.date(), val_end.date())

    # 地下鉄側: all/time_wdAM/time_weMID それぞれで analyze_segment を実行。
    subway_rows = {}
    for seg in sorted(merged["segment"].unique()):
        sub = merged[merged["segment"] == seg]
        res = analyze_segment(sub, analyze_cfg)
        res["corr_precipitation_hours"] = round(res["corr_precipitation_hours"], 3)
        subway_rows[seg] = res
        logger.info(
            "subway/%s: n_days=%d mean_per_day=%d W=%+.3f delta_r2=%+.3f",
            seg, res["n_days"], res["mean_per_day"],
            res["corr_precipitation_hours"], res["delta_r2"],
        )

    # 自転車側(Phase A)は再計算しない。既存 results/segment_weather_sensitivity.csv から読む。
    bike_df = pd.read_csv(PROJECT_ROOT / "results" / "segment_weather_sensitivity.csv")
    bike_df = bike_df.set_index("segment")

    rows = []
    for seg_key, seg_label in SEGMENT_LABELS.items():
        rows.append({
            "mode": "bike",
            "segment": seg_label,
            "W_precip": float(bike_df.loc[seg_key, "corr_precipitation_hours"]),
            "delta_r2": float(bike_df.loc[seg_key, "delta_r2"]),
        })
        rows.append({
            "mode": "subway",
            "segment": seg_label,
            "W_precip": subway_rows[seg_key]["corr_precipitation_hours"],
            "delta_r2": subway_rows[seg_key]["delta_r2"],
        })

    df = pd.DataFrame(rows)
    dest = PROJECT_ROOT / cfg["phase_b"]["paths"]["results"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    logger.info("Saved %s", dest)

    print(df.to_string(index=False))

    piv_w = df.pivot(index="mode", columns="segment", values="W_precip") \
              .reindex(index=["bike", "subway"], columns=["commute", "leisure"])
    piv_r = df.pivot(index="mode", columns="segment", values="delta_r2") \
              .reindex(index=["bike", "subway"], columns=["commute", "leisure"])

    print("\n=== 2x2 (W = corr(残差, 降水時間アノマリ); 負ほど雨に弱い) ===")
    print(piv_w.to_string())
    print("\n=== 2x2 (delta_r2 = 天候追加による test R^2 の増分) ===")
    print(piv_r.to_string())

    print("\n=== 読み方（事前にどちらが正しいと決めつけず、数値をそのまま提示） ===")
    print("-- 列（segment: commute vs leisure）の差 = 裁量性の影響 --")
    col_effect = {}
    for mode in ["bike", "subway"]:
        w_c = piv_w.loc[mode, "commute"]
        w_l = piv_w.loc[mode, "leisure"]
        diff = w_l - w_c
        col_effect[mode] = diff
        direction = "レジャー側がより雨に弱い" if diff < 0 else "通勤側の方が雨に弱い、または同程度"
        print(f"  {mode}: W[commute]={w_c:+.3f}  W[leisure]={w_l:+.3f}  "
              f"差(leisure-commute)={diff:+.3f}  -> {direction}")

    print("\n-- 行（mode: bike vs subway、同一segment内）の差 = 曝露・モード代替の影響 --")
    row_effect = {}
    for seg in ["commute", "leisure"]:
        w_b = piv_w.loc["bike", seg]
        w_s = piv_w.loc["subway", seg]
        diff = w_s - w_b
        row_effect[seg] = diff
        print(f"  {seg}: W[bike]={w_b:+.3f}  W[subway]={w_s:+.3f}  差(subway-bike)={diff:+.3f}")

    print("\n-- 総合（ニュートラル・判定は書くが結論の断定はしない） --")
    same_dir = (col_effect["bike"] < 0) == (col_effect["subway"] < 0)
    print(f"  列効果(leisure-commute): bike={col_effect['bike']:+.3f}, subway={col_effect['subway']:+.3f}"
          f"  -> 両modeで方向が{'一致' if same_dir else '不一致'}")
    print(f"  行効果(subway-bike): commute内={row_effect['commute']:+.3f}, leisure内={row_effect['leisure']:+.3f}")
    max_col = max(abs(col_effect["bike"]), abs(col_effect["subway"]))
    max_row = max(abs(row_effect["commute"]), abs(row_effect["leisure"]))
    print(f"  |列効果|の最大={max_col:.3f}  vs  |行効果|の最大={max_row:.3f}")
    if same_dir:
        print("  -> 列（裁量性=commute/leisure）の効果が両modeで同方向であり、裁量性仮説を支持する材料の一つ。")
    else:
        print("  -> 列（裁量性）の効果がmode間で方向不一致であり、裁量性だけでは説明しきれない可能性がある。")
    if max_row > max_col:
        print("  -> 行（曝露・モード代替）の効果の方が列（裁量性）より大きく、曝露/代替の交絡が無視できない。")
    else:
        print("  -> 行（曝露・モード代替）の効果は列（裁量性）ほど大きくない。")

    n_bike = int(bike_df.loc["time_wdAM", "n_days"])  # bike/subway で n_days が異なるための限界注記
    n_subway_wdam = subway_rows["time_wdAM"]["n_days"]
    print(f"\n[限界] bikeは2022-2023年、subwayは2023-2024年と対象期間が異なる（アノマリ化で季節は吸収するが期間ズレは残る）。"
          f" n_days も bike time_wdAM={n_bike}日 vs subway time_wdAM={n_subway_wdam}日 のように異なり、"
          f"薄いセルはこのプロジェクトの既定方針として許容する。")


if __name__ == "__main__":
    main()
