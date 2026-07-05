"""
make_figures.py — ポスター用図版をmatplotlibで生成する。

出力: poster/figures/fig1_taxi_bike_bars.png
      poster/figures/fig2_discretion_axes.png
      poster/figures/fig3_2x2_heatmap.png
      poster/figures/concept_residual_flow.png

配色は3色(+白黒グレー)に固定: TEAL(有意・強反応) / BRICK(逆向き・要注意) / GRAY(非有意・不使用)。
数値は全て研究の実測値（rehab/step2_inference.py, rehab/step3_phaseb_inference.py の出力）。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1B2420"
PAPER = "#FFFFFF"
TEAL = "#1F6F78"
BRICK = "#B54834"
GRAY = "#9CA3AF"
DPI = 320


def _style_ax(ax, hide_y_spine=True):
    ax.set_facecolor(PAPER)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if hide_y_spine:
        ax.spines["left"].set_visible(False)
    ax.tick_params(colors=INK, labelsize=13)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK)


# ---------------------------------------------------------------------------
# 図1: タクシー vs 自転車 対比棒グラフ
# ---------------------------------------------------------------------------
def fig1_taxi_bike_bars():
    labels = ["タクシー", "自転車"]
    point = [-0.024, -0.515]
    lo = [-0.099, -0.641]
    hi = [+0.074, -0.445]
    sig = [False, True]

    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=DPI)
    fig.patch.set_facecolor(PAPER)

    colors = [TEAL if s else GRAY for s in sig]
    xerr = np.array([[p - l for p, l in zip(point, lo)],
                     [h - p for p, h in zip(point, hi)]])

    ax.barh(y, point, height=0.42, color=colors, zorder=3,
            xerr=xerr, error_kw=dict(ecolor=INK, elinewidth=1.6, capsize=6, capthick=1.6, zorder=4))
    ax.axvline(0, color=INK, linewidth=1.2, zorder=2)

    # 数値ラベルはバーの真上（誤差バーと重ならない高さ）に置く
    for yi, p in zip(y, point):
        ax.text(p, yi + 0.34, f"{p:+.3f}", va="bottom", ha="center",
                fontsize=17, fontweight="bold", color=INK, zorder=6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=17)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.65, 1.65)
    ax.set_xlim(-0.75, 0.2)
    ax.set_xlabel("連動係数（負=雨で減る）", fontsize=14, color=INK, labelpad=10)
    ax.set_title("降水量への反応の強さ（残差との連動係数）", fontsize=18, color=INK, pad=18, fontweight="bold")
    _style_ax(ax, hide_y_spine=True)
    ax.spines["bottom"].set_color(INK)

    handles = [plt.Rectangle((0, 0), 1, 1, color=TEAL), plt.Rectangle((0, 0), 1, 1, color=GRAY)]
    ax.legend(handles, ["有意（信頼区間が0を含まない）", "誤差の範囲（信頼区間が0をまたぐ）"],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False,
              fontsize=12, labelcolor=INK)

    fig.savefig(OUT / "fig1_taxi_bike_bars.png", facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 図2: 裁量性仮説 3軸検証パネル
# ---------------------------------------------------------------------------
def fig2_discretion_axes():
    # (軸名, 左ラベル, 右ラベル, マーカー位置0-1(1=右=より雨に弱い), 判定, 判定色)
    rows = [
        ("軸1 往復/片道", "片道(通勤)", "往復(レジャー)", 0.30, "✗", BRICK, "仮説と逆"),
        ("軸2 時間帯", "平日朝(通勤)", "休日昼(レジャー)", 0.78, "✓", TEAL, "支持"),
        ("軸3 会員種別", "会員(通勤寄り)", "非会員(レジャー寄り)", None, "△", GRAY, "検証に使えず"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=DPI)
    fig.patch.set_facecolor(PAPER)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(rows))
    ax.axis("off")

    track_x0, track_x1 = 3.0, 7.6
    for i, (name, left_lbl, right_lbl, pos, badge, color, note) in enumerate(rows):
        yc = len(rows) - i - 0.5
        ax.text(0.1, yc + 0.28, name, fontsize=16, fontweight="bold", color=INK, va="center")

        track_color = GRAY if pos is None else INK
        ax.plot([track_x0, track_x1], [yc, yc], color=track_color, linewidth=2.4,
                 alpha=(0.35 if pos is None else 1.0), zorder=2)
        ax.text(track_x0, yc - 0.32, left_lbl, fontsize=12.5, color=INK, ha="left", va="top")
        ax.text(track_x1, yc - 0.32, right_lbl, fontsize=12.5, color=INK, ha="right", va="top")

        if pos is not None:
            mx = track_x0 + pos * (track_x1 - track_x0)
            ax.scatter([mx], [yc], s=340, color=color, zorder=5, edgecolor=INK, linewidth=1.2)
        else:
            mx = (track_x0 + track_x1) / 2
            ax.scatter([mx], [yc], s=340, color=GRAY, zorder=5, edgecolor=INK,
                       linewidth=1.2, alpha=0.5)

        ax.text(9.3, yc, badge, fontsize=30, color=color, ha="center", va="center", fontweight="bold")
        ax.text(9.3, yc - 0.42, note, fontsize=10.5, color=INK, ha="center", va="top")

    ax.set_title("裁量性仮説の3軸検証", fontsize=19, color=INK, pad=14, fontweight="bold", x=0.36)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_discretion_axes.png", facecolor=PAPER)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 図3: 交通手段×移動目的 2×2ヒートマップ（★主役）
# ---------------------------------------------------------------------------
def fig3_2x2_heatmap():
    values = np.array([[-0.451, -0.667],
                       [-0.078, -0.465]])
    sig = np.array([[True, True],
                    [False, True]])
    row_labels = ["自転車\n(濡れる)", "地下鉄\n(濡れない)"]
    col_labels = ["通勤", "レジャー"]

    cmap = mcolors.LinearSegmentedColormap.from_list("white_teal", ["#FFFFFF", TEAL])
    mag = np.abs(values)
    norm = mcolors.Normalize(vmin=0, vmax=0.75)

    fig, ax = plt.subplots(figsize=(6.6, 6.2), dpi=DPI)
    fig.patch.set_facecolor(PAPER)
    im = ax.imshow(mag, cmap=cmap, norm=norm, aspect="equal")

    for i in range(2):
        for j in range(2):
            v = values[i, j]
            text_color = "#FFFFFF" if mag[i, j] > 0.42 else INK
            star = "  ★" if sig[i, j] else ""
            ax.text(j, i - 0.06, f"{v:+.3f}{star}", ha="center", va="center",
                    fontsize=22, fontweight="bold", color=text_color)
            if not sig[i, j]:
                ax.text(j, i + 0.30, "※データ少・慎重に", ha="center", va="center",
                        fontsize=10.5, color=BRICK, fontweight="bold")

    ax.set_xticks([0, 1]); ax.set_xticklabels(col_labels, fontsize=16, color=INK)
    ax.set_yticks([0, 1]); ax.set_yticklabels(row_labels, fontsize=14.5, color=INK)
    ax.tick_params(length=0)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color=PAPER, linewidth=4)
    ax.tick_params(which="minor", length=0)

    ax.set_title("交通手段×移動目的でみた降水への反応（係数W）", fontsize=15.5, color=INK, pad=54, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_2x2_heatmap.png", facecolor=PAPER)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 概念図: 残差化フロー（① 生データ → ② パターン除去 → ③ 残差×降水 照合）
# ---------------------------------------------------------------------------
def concept_residual_flow():
    rng = np.random.default_rng(7)
    t = np.linspace(0, 4 * np.pi, 120)
    trend = 1.0 + 0.15 * t
    season = 0.8 * np.sin(t) + 0.25 * np.sin(3 * t)
    noise = rng.normal(0, 0.12, len(t))
    raw = trend + season + noise
    baseline = trend + season
    resid = raw - baseline
    rain = np.clip(rng.gamma(1.2, 0.5, len(t)) - 0.3, 0, None)
    resid_vs_rain = -0.6 * rain + rng.normal(0, 0.25, len(t))

    fig = plt.figure(figsize=(12.5, 4.4), dpi=DPI)
    fig.patch.set_facecolor(PAPER)
    gs = fig.add_gridspec(1, 3, wspace=0.38, left=0.04, right=0.98, top=0.78, bottom=0.2)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    ax = axes[0]
    ax.plot(t, raw, color=INK, linewidth=1.8)
    ax.set_title("① 生データ", fontsize=15, color=INK, fontweight="bold")

    ax = axes[1]
    ax.plot(t, raw, color=GRAY, linewidth=1.4, label="生データ")
    ax.plot(t, baseline, color=TEAL, linewidth=2.2, label="いつものパターン")
    for i in range(0, len(t), 14):
        ax.plot([t[i], t[i]], [baseline[i], raw[i]], color=BRICK, linewidth=1.0, alpha=0.7)
    ax.set_title("② パターンを引く（残差化）", fontsize=15, color=INK, fontweight="bold")
    ax.legend(fontsize=10, frameon=False, loc="upper left", labelcolor=INK)

    ax = axes[2]
    ax.scatter(rain, resid_vs_rain, s=22, color=TEAL, alpha=0.75, edgecolor="none")
    zz = np.polyfit(rain, resid_vs_rain, 1)
    xs = np.linspace(rain.min(), rain.max(), 10)
    ax.plot(xs, np.polyval(zz, xs), color=BRICK, linewidth=2.0)
    ax.set_xlabel("降水", fontsize=12, color=INK)
    ax.set_ylabel("残差", fontsize=12, color=INK)
    ax.set_title("③ 残差と降水を照合", fontsize=15, color=INK, fontweight="bold")

    for ax in axes:
        ax.set_facecolor(PAPER)
        ax.set_xticks([]);
        if ax is not axes[2]:
            ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(INK); s.set_linewidth(1.0)

    # 矢印（サブプロット間）
    for i in (0, 1):
        p0 = axes[i].get_position()
        p1 = axes[i + 1].get_position()
        arrow = FancyArrowPatch((p0.x1 + 0.005, (p0.y0 + p0.y1) / 2),
                                (p1.x0 - 0.005, (p1.y0 + p1.y1) / 2),
                                transform=fig.transFigure, arrowstyle="-|>",
                                mutation_scale=22, color=INK, linewidth=1.6)
        fig.patches.append(arrow)

    fig.suptitle("「いつものパターンを引いて、残ったブレを見る」", fontsize=17, color=INK, fontweight="bold", y=0.97)
    fig.savefig(OUT / "concept_residual_flow.png", facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    fig1_taxi_bike_bars()
    fig2_discretion_axes()
    fig3_2x2_heatmap()
    concept_residual_flow()
    print("saved to", OUT)
