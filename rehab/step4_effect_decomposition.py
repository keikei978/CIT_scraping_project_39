"""
ステップ4（ポスター用・効果分解のCI): 列効果（裁量性）と行効果（曝露）を
ブロック・ブートストラップで直接CI付きにする。4セルは全て独立（日付が重ならない）
なので、各セルのブートストラップ複製をindex単位で引き算すれば、差の分布として妥当。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _moving_block_indices
from step3_phaseb_inference import phaseA_seg, phaseB_seg, residualize

NBOOT = 5000
BLOCK = 14


def boot_w(resid, anom, seed):
    n = len(resid)
    rng = np.random.default_rng(seed)
    point = float(np.corrcoef(anom, resid)[0, 1])
    boots = np.empty(NBOOT)
    for b in range(NBOOT):
        idx = _moving_block_indices(n, BLOCK, rng)
        a, r = anom[idx], resid[idx]
        boots[b] = np.corrcoef(a, r)[0, 1] if np.std(a) > 0 and np.std(r) > 0 else np.nan
    return point, boots


def report(label, arr):
    lo, hi = np.nanpercentile(arr, [2.5, 97.5])
    sig = lo > 0 or hi < 0
    print(f"  {label:24s} point={np.nanmean(arr):+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  "
          f"{'有意' if sig else '★有意でない'}")
    return lo, hi, sig


def main():
    bc_r, bc_a = residualize(phaseA_seg("time_wdAM"), "2023-06-30")
    bl_r, bl_a = residualize(phaseA_seg("time_weMID"), "2023-06-30")
    sc_r, sc_a = residualize(phaseB_seg("time_wdAM"), "2024-05-31")
    sl_r, sl_a = residualize(phaseB_seg("time_weMID"), "2024-05-31")

    bc_p, bc_b = boot_w(bc_r, bc_a, seed=1)
    bl_p, bl_b = boot_w(bl_r, bl_a, seed=2)
    sc_p, sc_b = boot_w(sc_r, sc_a, seed=3)
    sl_p, sl_b = boot_w(sl_r, sl_a, seed=4)

    print("4セル点推定: bike-commute=%.3f bike-leisure=%.3f subway-commute=%.3f subway-leisure=%.3f"
          % (bc_p, bl_p, sc_p, sl_p))

    print("\n列効果（裁量性: レジャー-通勤）")
    col_bike = bl_b - bc_b
    col_sub = sl_b - sc_b
    report("列効果(bike)", col_bike)
    report("列効果(subway)", col_sub)

    print("\n行効果（曝露: 地下鉄-自転車）")
    row_commute = sc_b - bc_b
    row_leisure = sl_b - bl_b
    report("行効果(commute)", row_commute)
    report("行効果(leisure)", row_leisure)


if __name__ == "__main__":
    main()
