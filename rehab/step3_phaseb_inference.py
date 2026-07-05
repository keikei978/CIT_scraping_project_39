"""
ステップ3（Phase B検定）: 「列効果（裁量性）と行効果（曝露）がほぼ同じ大きさ」という
Phase B の要約主張を、rehab の既存手法（ブロック・ブートストラップCI + 自己相関補正Fisher-z）
でそのまま検定する。

対象4セル（W=corr(残差, 降水時間アノマリ)）:
  bike-commute (time_wdAM, nyc-weather-discretion Phase A, train_end=2023-06-30)
  bike-leisure (time_weMID, 同上)
  subway-commute (time_wdAM, nyc-weather-discretion Phase B, train_end=2024-05-31)
  subway-leisure (time_weMID, 同上)

4セルは全て日付が重ならない（bike=2022-23 vs subway=2023-24、time_wdAM=平日 vs time_weMID=休日）
ため、すべて独立2群として Fisher-z（有効n補正）で比較する。paired bootstrap は使わない
（rehab step2 の「軸2 時間帯(独立)」と同じ扱い）。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (fit_baseline, weather_anomaly, block_bootstrap_corr,
                    effective_n, fisher_z_diff)

BASE = Path("C:/Users/keisu/Documents/projects/CIT_Assignment/scraping_project")
NH = 3
NBOOT = 5000
BLOCK = 14
WVAR = "precipitation_hours"


def phaseA_seg(segment):
    d = pd.read_parquet(BASE / "nyc-weather-discretion/data/merged.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["segment"] == segment].sort_values("date").reset_index(drop=True)
    return d.rename(columns={"n": "y"})


def phaseB_seg(segment):
    d = pd.read_parquet(BASE / "nyc-weather-discretion/data/phaseB_merged.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["segment"] == segment].sort_values("date").reset_index(drop=True)
    return d.rename(columns={"n": "y"})


def residualize(df, train_end, wvar=WVAR):
    dates = pd.to_datetime(df["date"])
    tmask = (dates <= pd.Timestamp(train_end)).to_numpy()
    resid = df["y"].to_numpy().astype(float) - fit_baseline(df, tmask, "y", NH)
    anom = weather_anomaly(df, tmask, wvar, NH)
    return resid, anom


def cell(label, df, train_end):
    resid, anom = residualize(df, train_end)
    ci = block_bootstrap_corr(anom, resid, block=BLOCK, n_boot=NBOOT)
    ne = effective_n(resid)
    flag = "0除外(有意)" if ci["p_excludes_0"] else "0を含む(有意でない)"
    print(f"  {label:16s} W={ci['point']:+.3f}  95%CI[{ci['ci_lo']:+.3f},{ci['ci_hi']:+.3f}]  "
          f"n={len(resid)} n_eff={ne:.0f}  {flag}")
    return {"W": ci["point"], "n_eff": ne, "resid": resid, "anom": anom}


def diff_test(labelA, cA, labelB, cB, hypothesis_note=""):
    fz = fisher_z_diff(cA["W"], cA["n_eff"], cB["W"], cB["n_eff"])
    v = "有意" if fz["p"] < 0.05 else "★有意でない"
    print(f"  {labelA} - {labelB} = {cA['W']-cB['W']:+.3f}  "
          f"Fisher-z p={fz['p']:.4f}  {v}  {hypothesis_note}")
    return fz


def main():
    print("=" * 90)
    print("Phase B の4セル: W=corr(残差, 降水時間アノマリ) ブロック・ブートストラップCI")
    print("=" * 90)
    bike_c = cell("bike-commute", phaseA_seg("time_wdAM"), "2023-06-30")
    bike_l = cell("bike-leisure", phaseA_seg("time_weMID"), "2023-06-30")
    sub_c = cell("subway-commute", phaseB_seg("time_wdAM"), "2024-05-31")
    sub_l = cell("subway-leisure", phaseB_seg("time_weMID"), "2024-05-31")

    print("\n" + "=" * 90)
    print("列効果（裁量性: レジャー-通勤, 同一モード内 → 独立2群 Fisher-z）")
    print("=" * 90)
    col_bike = diff_test("bike-leisure", bike_l, "bike-commute", bike_c,
                         "(負なら裁量性仮説を支持)")
    col_sub = diff_test("subway-leisure", sub_l, "subway-commute", sub_c,
                        "(負なら裁量性仮説を支持)")

    print("\n" + "=" * 90)
    print("行効果（曝露・代替: 地下鉄-自転車, 同一時間帯内 → 独立2群 Fisher-z）")
    print("=" * 90)
    row_commute = diff_test("subway-commute", sub_c, "bike-commute", bike_c,
                            "(正なら地下鉄の方が雨に強い=曝露の効果)")
    row_leisure = diff_test("subway-leisure", sub_l, "bike-leisure", bike_l,
                            "(正なら地下鉄の方が雨に強い=曝露の効果)")

    print("\n" + "=" * 90)
    print("「列効果 ≈ 行効果」という要約主張の検証")
    print("=" * 90)
    col_vals = {"bike": bike_l["W"] - bike_c["W"], "subway": sub_l["W"] - sub_c["W"]}
    row_vals = {"commute": sub_c["W"] - bike_c["W"], "leisure": sub_l["W"] - bike_l["W"]}
    print(f"  列効果: bike={col_vals['bike']:+.3f}  subway={col_vals['subway']:+.3f}")
    print(f"  行効果: commute={row_vals['commute']:+.3f}  leisure={row_vals['leisure']:+.3f}")
    print(f"  |列効果|最大={max(abs(v) for v in col_vals.values()):.3f}  "
          f"|行効果|最大={max(abs(v) for v in row_vals.values()):.3f}")
    print("\n  [重要な限界] 上の『列と行、どちらの絶対値が大きいか』という比較そのものについては、")
    print("  ここでは個別の有意性検定(vs 0)しか行っていない。『列効果と行効果が統計的に等しい』")
    print("  という主張自体を検定するには、両者の差(交互作用)のCIを別途出す必要があり、それは")
    print("  未実施。したがって『ほぼ同じ大きさ』は点推定同士が近いという記述であって、")
    print("  統計的に有意な等価性を主張するものではない。")


if __name__ == "__main__":
    main()
