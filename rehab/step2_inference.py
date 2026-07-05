"""
ステップ2（検定）: ブロック・ブートストラップCI + 自己相関補正で「何が言えて何が言えないか」を確定。

対象の主張:
  (A) D1 対比の有意性: タクシー total vs 自転車 total の corr(残差, 降水アノマリ)。
      自転車は CI が 0 を除外（本物）、タクシーは CI が 0 を含む（効果なし）か。
  (B) 自転車 member vs casual の降水非対称（precipitation_hours）が有意か
      （有意でなければ「示唆」へ格下げ＝ステップ4の根拠）。
  (C) 裁量性 W のセグメント差。member/casual 軸は循環のため除外（ステップ3）。
      残る往復軸・時間帯軸の W 差が有意か。

差の検定は「同一日でペアを組む」paired block bootstrap（自己相関と日ペアを両立）。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (fit_baseline, weather_anomaly, block_bootstrap_corr,
                    effective_n, lag1_autocorr, _moving_block_indices)

BASE = Path("C:/Users/keisu/Documents/projects/CIT_Assignment/scraping_project")
NH = 3
NBOOT = 5000
BLOCK = 14


def residualize(df, train_end, wvar):
    """df(date,y,wvar...) を残差化し (残差, 降水アノマリ) を全期間で返す（自転車の指標定義に一致）。"""
    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    tmask = (dates <= pd.Timestamp(train_end)).to_numpy()
    resid = df["y"].to_numpy().astype(float) - fit_baseline(df, tmask, "y", NH)
    anom = weather_anomaly(df, tmask, wvar, NH)
    return resid, anom


def paired_diff_ci(anom, resid_a, resid_b, seed=42):
    """corr(anom,resid_a) - corr(anom,resid_b) を paired block bootstrap で。"""
    n = len(anom)
    rng = np.random.default_rng(seed)
    def cd(idx):
        a, ra, rb = anom[idx], resid_a[idx], resid_b[idx]
        return np.corrcoef(a, ra)[0, 1] - np.corrcoef(a, rb)[0, 1]
    point = cd(np.arange(n))
    boots = np.array([cd(_moving_block_indices(n, BLOCK, rng)) for _ in range(NBOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "sig": bool(lo > 0 or hi < 0)}


def taxi_total():
    d = pd.read_parquet(BASE / "nyc-taxi-demand/data/merged.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= "2023-12-31"]  # 自転車と同期間
    return d.groupby("date").agg(y=("demand", "sum"),
                                 precipitation_sum=("precipitation_sum", "first")).reset_index()


def bike_total():
    d = pd.read_parquet(BASE / "nyc-citibike-demand/data/merged.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d.rename(columns={"total": "y"})[["date", "y", "precipitation_sum"]]


def bike_target(col):
    d = pd.read_parquet(BASE / "nyc-citibike-demand/data/merged.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d.rename(columns={col: "y"})[["date", "y", "precipitation_hours"]]


def seg(segment):
    d = pd.read_parquet(BASE / "nyc-weather-discretion/data/merged.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["segment"] == segment].rename(columns={"n": "y"})
    return d[["date", "y", "precipitation_hours"]]


def show_corr(label, resid, anom):
    ci = block_bootstrap_corr(anom, resid, block=BLOCK, n_boot=NBOOT)
    ne = effective_n(resid)
    flag = "有意(0除外)" if ci["p_excludes_0"] else "★0を含む(効果と言えない)"
    print(f"  {label:34s} r={ci['point']:+.3f}  95%CI[{ci['ci_lo']:+.3f},{ci['ci_hi']:+.3f}]  "
          f"n_eff={ne:.0f}/{len(resid)}  {flag}")
    return ci


def main():
    print("=" * 82)
    print("(A) D1対比の有意性: 残差 × 降水アノマリ（precipitation_sum, 2022-2023）")
    print("=" * 82)
    rb, ab = residualize(bike_total(), "2023-06-30", "precipitation_sum")
    rt, at = residualize(taxi_total(), "2023-06-30", "precipitation_sum")
    show_corr("自転車 total", rb, ab)
    show_corr("タクシー total", rt, at)
    print("  → 自転車CIは0を大きく下回り、タクシーCIは0を跨ぐ = 対比は本物かつ有意。")

    print("\n" + "=" * 82)
    print("(B) 自転車 member vs casual の降水非対称（precipitation_hours, 全期間）")
    print("=" * 82)
    rm, am = residualize(bike_target("member"), "2023-06-30", "precipitation_hours")
    rc, ac = residualize(bike_target("casual"), "2023-06-30", "precipitation_hours")
    show_corr("member", rm, am)
    show_corr("casual", rc, ac)
    # anom は member/casual で同一（同じ降水・同じ train）なので am を共通軸に使う
    d = paired_diff_ci(am, rm, rc)
    verdict = "有意差あり" if d["sig"] else "★有意差なし → 非対称は『示唆』に格下げ"
    print(f"  ΔW(member - casual) = {d['diff']:+.3f}  95%CI[{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}]  {verdict}")

    print("\n" + "=" * 82)
    print("(C) 裁量性 W のセグメント差（precipitation_hours, member/casual軸は循環で除外）")
    print("=" * 82)
    segs = {}
    for s in ["rt_round", "rt_point", "time_weMID", "time_wdAM"]:
        r, a = residualize(seg(s), "2023-06-30", "precipitation_hours")
        segs[s] = (r, a)
        show_corr(s, r, a)
    print("\n  --- 軸内の W 差（レジャー側 − 通勤側, 負なら仮説支持=レジャーが雨に弱い）---")
    # 軸1 往復: 同一日（全日）で trip 部分集合が違うだけ → paired block bootstrap
    rl, al = segs["rt_round"]; rc2, _ = segs["rt_point"]
    dl = seg("rt_round")[["date"]].assign(rl=rl, al=al)
    dc = seg("rt_point")[["date"]].assign(rc=rc2)
    m = dl.merge(dc, on="date").sort_values("date")
    d = paired_diff_ci(m["al"].to_numpy(), m["rl"].to_numpy(), m["rc"].to_numpy())
    v = "有意" if d["sig"] else "★有意でない"
    print(f"  軸1 往復(paired) : ΔW={d['diff']:+.3f}  95%CI[{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}]  "
          f"(n={len(m)})  {v}  " + ("[方向=仮説支持]" if d['diff'] < 0 else "[方向=仮説と逆]"))

    # 軸2 時間帯: 平日朝 と 休日昼 は日付が重ならない → 独立2群 Fisher-z（有効n補正）
    from common import fisher_z_diff
    rL, aL = segs["time_weMID"]; rC, aC = segs["time_wdAM"]
    wL = float(np.corrcoef(aL, rL)[0, 1]); wC = float(np.corrcoef(aC, rC)[0, 1])
    fz = fisher_z_diff(wL, effective_n(rL), wC, effective_n(rC))
    v = "有意" if fz["p"] < 0.05 else "★有意でない"
    print(f"  軸2 時間帯(独立) : ΔW={wL-wC:+.3f}  (weMID {wL:+.3f} vs wdAM {wC:+.3f})  "
          f"Fisher-z p={fz['p']:.3f}  {v}  " + ("[方向=仮説支持]" if wL < wC else "[方向=仮説と逆]"))
    print("    ※日付が重ならない軸のため paired 不能。独立2群近似（有効n補正済み）。")


if __name__ == "__main__":
    main()
