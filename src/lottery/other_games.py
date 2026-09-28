"""威力彩/大樂透以外的台彩遊戲：理論回收率（依官方獎金表）＋實際回收率（官方 API 每期銷售額 vs 實發獎金）。
先跑 python src/lottery/fetch.py d539 m539 m649 d3 d4；執行：uv run --with numpy python src/lottery/other_games.py"""
import json
from math import comb
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "lottery"
TAX_FREE, KEEP = 5000, 0.796  # 單筆 > 5,000 元扣 20% 所得稅＋0.4% 印花稅


def net(prize):
    return prize if prize <= TAX_FREE else prize * KEEP


def hyper(n, k, draw, pool):
    """選 n 個號碼、開出 draw 個（共 pool 個）時剛好中 k 個的機率。"""
    return comb(draw, k) * comb(pool - draw, n - k) / comb(pool, n)


# 賓果賓果官方倍數表（每注 25 元；taiwanlottery.com/lotto/info/bingo_bingo，2026-09 讀取）
BINGO = {10: {10: 200000, 9: 10000, 8: 1000, 7: 100, 6: 10, 5: 1, 0: 1},
         9: {9: 40000, 8: 4000, 7: 120, 6: 20, 5: 4, 4: 1, 0: 1},
         8: {8: 20000, 7: 800, 6: 40, 5: 8, 4: 1, 0: 1},
         7: {7: 3200, 6: 120, 5: 12, 4: 2, 3: 1},
         6: {6: 1000, 5: 40, 4: 8, 3: 1},
         5: {5: 300, 4: 20, 3: 2},
         4: {4: 40, 3: 4, 2: 1},
         3: {3: 20, 2: 2},
         2: {2: 3, 1: 1},
         1: {1: 2}}
# 2026 春節加碼（新聞報導的 6 項；另 3 項未查到，保守不計）
BINGO_PROMO = {1: {1: 3}, 2: {2: 6}, 3: {3: 40}, 4: {4: 80}, 5: {5: 400}, 6: {6: 2000}}


def bingo_ev(stars, table):
    return sum(hyper(stars, k, 20, 80) * net(m * 25) for k, m in table.items()) / 25


def fixed_ev():
    """固定獎金遊戲每注稅後回收率。"""
    out = {}
    for s in range(1, 11):
        promo = {**BINGO[s], **BINGO_PROMO.get(s, {})}
        out[f"賓果 {s}星"] = (bingo_ev(s, BINGO[s]), bingo_ev(s, promo) if s in BINGO_PROMO else None)
    out["賓果 超級獎號"] = (net(1200) / 80 / 25, None)
    p13 = sum(comb(40, k) * comb(40, 20 - k) for k in range(13, 21)) / comb(80, 20)
    out["賓果 猜大小/單雙"] = (p13 * 150 / 25, None)
    for name, pool, draw, prizes in (("39樂合彩", 39, 5, {2: 1125, 3: 11250, 4: 212500}),
                                     ("49樂合彩", 49, 6, {2: 1250, 3: 12500, 4: 200000})):
        for n, pz in prizes.items():
            out[f"{name} {n}合"] = (hyper(n, n, draw, pool) * net(pz) / 25, None)
    out["39樂合彩 2合"] = (out["39樂合彩 2合"][0], hyper(2, 2, 5, 39) * 1500 / 25)  # 歷年加碼 1,500 元
    # 3/4星彩 2024 新制：由個位數往前連續對中；中高獎不再領低獎（實際回收率吻合此假設）
    out["3星彩"] = ((0.09 * 50 + 0.009 * 500 + 0.001 * 5000) / 25, None)
    out["4星彩"] = ((0.009 * 500 + 0.0009 * net(5000) + 0.0001 * net(50000)) / 25, None)
    # 今彩539：貳參肆獎固定；頭獎 800 萬（最多 3 人各 800 萬，之後 2,400 萬均分）
    P = {k: hyper(5, k, 5, 39) for k in (5, 4, 3, 2)}
    out["今彩539"] = ((P[5] * net(8e6) + P[4] * net(20000) + P[3] * 300 + P[2] * 50) / 50,
                      (P[5] * net(1e7) + P[4] * net(20000) + P[3] * 300 + P[2] * 50) / 50)  # 頭獎加碼 1,000 萬期間
    return out


def realized():
    out = {}
    for g, name in (("d539", "今彩539"), ("m539", "39樂合彩"), ("m649", "49樂合彩"), ("d3", "3星彩"), ("d4", "4星彩")):
        rows = json.loads((DATA / f"{g}.json").read_text(encoding="utf-8"))
        for label, sub in (("全期", rows), ("2024起", [r for r in rows if r["lotteryDate"] >= "2024"])):
            s, t = sum(r["sellAmount"] for r in sub), sum(r["totalAmount"] for r in sub)
            out[f"{name} {label}"] = (t / s, len(sub))
    return out


if __name__ == "__main__":
    ev = fixed_ev()
    assert abs(bingo_ev(1, BINGO[1]) - 0.5) < 1e-9  # 1星：1/4 × 2 倍
    assert abs(ev["39樂合彩 2合"][0] - 10 / 741 * 45) < 1e-9
    print("理論稅後回收率（平時 / 加碼時）")
    for k, (a, b) in sorted(ev.items(), key=lambda x: -x[1][0]):
        print(f"  {k:12s} {a:6.1%}" + (f"  加碼 {b:6.1%}" if b else ""))
    print("實際回收率（稅前，全體彩民合計）")
    for k, (r, n) in realized().items():
        print(f"  {k:14s} {r:6.1%}  ({n} 期)")
