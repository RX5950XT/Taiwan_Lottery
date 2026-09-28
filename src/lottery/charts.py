"""讀 output/analysis.json 畫報告用圖 → output/*.png
執行：uv run --with matplotlib,numpy python src/lottery/charts.py"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"
A = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))
BLUE, ORANGE, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#fcfcfb"
NAMES = {"power": "威力彩（第一區 1–38）", "lotto": "大樂透（1–49）"}
plt.rcParams.update({"font.family": ["Microsoft JhengHei", "sans-serif"], "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e3",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.unicode_minus": False})


def freq_chart():
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
    for ax, g in zip(axes, ("power", "lotto")):
        R = A[g]["randomness"]
        f, e, n = np.array(R["freq"]), R["expected_freq"], len(R["freq"])
        sd = np.sqrt(R["draws"] * 6 / n * (1 - 6 / n))
        z = (f - e) / sd
        ax.bar(range(1, n + 1), z, color=BLUE, width=0.7)
        ax.axhspan(-2, 2, color="#f0efec", zorder=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlim(0.3, n + 0.7)
        ax.set_xticks(range(1, n + 1))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylabel("偏離程度 (z)")
        ax.set_title(f"{NAMES[g]}：{R['draws']} 期各號碼出現次數偏離  ·  卡方 p = {R['p_main']:.2f}",
                     loc="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "灰色帶 = ±2 標準差，純隨機下約 95% 的號碼會落在裡面。所有號碼都在帶內附近 → 沒有「熱號」。",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig_frequency.png", dpi=150)


def popularity_chart():
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
    for ax, g in zip(axes, ("power", "lotto")):
        c = np.array(A[g]["popularity"]["coef"])
        c = (np.exp(c) - 1) * 100  # 換成「比平均多幾 %」
        ax.bar(range(1, len(c) + 1), c, color=[ORANGE if v > 0 else BLUE for v in c], width=0.7)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlim(0.3, len(c) + 0.7)
        ax.set_xticks(range(1, len(c) + 1))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylabel("撞號人數 %")
        ax.set_title(f"{NAMES[g]}：開出這個號碼時，中獎人數比平均多/少幾 %", loc="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "橘 = 熱門（大家愛選，中了要跟很多人分）；藍 = 冷門。1–12（像月份、生日）明顯熱門，大號（威力彩 31 以上、大樂透 40 以上）明顯冷門。",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig_popularity.png", dpi=150)


def ev_chart():
    fig, ax = plt.subplots(figsize=(11, 5))
    for g, col, lab in (("power", BLUE, "威力彩"), ("lotto", ORANGE, "大樂透")):
        s = A[g]["ev"]["series"]
        ax.scatter(np.array(s["jackpot"]) / 1e8, s["ev_after_tax"], s=10, color=col, alpha=0.55,
                   label=lab, edgecolors="none")
    ax.axhline(1, color=INK, lw=1, ls="--")
    ax.text(0.2, 1.02, "回本線：每花 1 元期望拿回 1 元", fontsize=9, color=INK)
    ax.set_xscale("log")
    ticks = [0.2, 0.5, 1, 2, 5, 10, 20]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("當期頭獎池（億元，對數刻度）")
    ax.set_ylabel("每 1 元的稅後期望回收")
    ax.set_title("頭獎池越大，期望值越高；但 4,000 多期裡只有 16 期超過回本線", loc="left", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ev.png", dpi=150)


if __name__ == "__main__":
    freq_chart()
    popularity_chart()
    ev_chart()
    print("ok")
