"""運彩歷史賠率分析：抽成、各種固定押法的實際報酬、運彩盤 vs 國際盤的價差策略。
先跑 src/sports/scrape.py；執行：uv run --with numpy,scipy python src/sports/analyze.py → output/sports.json"""
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = ROOT / "data" / "sports", ROOT / "output"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def intl_line(cell):
    """國際盤口文字 → 等效盤口。『9贏』+『50%』：剛好 9 時押的一方贏一半 ≈ 8.75；『9輸』50% ≈ 9.25；『9』＝和局退款。"""
    if not cell or not cell["line"]:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)分?(贏|輸)?", cell["line"].replace("平", "0"))
    if not m:
        return None
    base = float(m.group(1))
    pct = float(cell["val"].rstrip("%")) / 100 if cell["val"].endswith("%") else 0
    return base - pct / 2 if m.group(2) == "贏" else base + pct / 2 if m.group(2) == "輸" else base


def load(league):
    games = []
    for line in (DATA / f"{league}.jsonl").read_text(encoding="utf-8").splitlines():
        g = json.loads(line)
        if "gid" in g:
            games.append(g)
    return list({g["gid"]: g for g in games}.values())


def bets(g):
    """展開成單注：(市場, 押的一方, 賠率, 報酬倍數[含本金；和局退款=1], 對手賠率, 附加資訊)。"""
    out, diff = [], g["hs"] - g["as"]
    a, h = (num(c["val"]) if c else None for c in g["ml"])
    if a and h and a > 1 and h > 1:
        for side, o, other, won in (("away", a, h, diff < 0), ("home", h, a, diff > 0)):
            out.append(("ml", side, o, 1.0 if diff == 0 else o * won, other, None))
    ca, ch = g["hcp"]
    la, oa, oh = (num(ca["line"]) if ca else None), (num(ca["val"]) if ca else None), (num(ch["val"]) if ch else None)
    if la is not None and oa and oh:
        m = -diff + la  # 客隊加上讓分後的淨勝分
        out.append(("hcp", "away", oa, 1.0 if m == 0 else oa * (m > 0), oh, la))
        out.append(("hcp", "home", oh, 1.0 if m == 0 else oh * (m < 0), oa, -la))
    co, cu = g["tot"]
    t, oo, ou = (num(co["line"]) if co else None), (num(co["val"]) if co else None), (num(cu["val"]) if cu else None)
    if t is not None and oo and ou:
        s = g["as"] + g["hs"]
        out.append(("tot", "over", oo, 1.0 if s == t else oo * (s > t), ou, t))
        out.append(("tot", "under", ou, 1.0 if s == t else ou * (s < t), oo, t))
    return out


def summarize(ret):
    ret = np.asarray(ret, float)
    if len(ret) < 30:
        return None
    t = stats.ttest_1samp(ret, 1.0, alternative="greater")
    return {"n": len(ret), "roi": float(ret.mean() - 1), "se": float(ret.std(ddof=1) / np.sqrt(len(ret))),
            "p_gt_breakeven": float(t.pvalue)}


def league_report(league):
    games = load(league)
    rows = [(g["date"][:4], *b) for g in games for b in bets(g)]
    rep = {"games": len(games), "years": sorted({r[0] for r in rows})}
    # 1. 抽成：每個市場「公平回收率」＝ 1 / (1/賠率A + 1/賠率B)
    rep["payout"] = {}
    for mk in ("ml", "hcp", "tot"):
        byyear = {}
        for y, m, side, o, r, other, _ in rows:
            if m == mk and side in ("away", "over"):
                byyear.setdefault(y, []).append(1 / (1 / o + 1 / other))
        rep["payout"][mk] = {y: round(float(np.mean(v)), 4) for y, v in sorted(byyear.items())}
    # 2. 固定押法
    strat = {}
    for y, m, side, o, r, other, line in rows:
        if m == "hcp":  # 讓分盤：讓分的一方＝熱門
            fav = "fav" if line < 0 else "dog"
        else:
            fav = "fav" if o < other else "dog" if o > other else "even"
        for key in (f"{m}:{side}", f"{m}:{fav}" if m != "tot" else f"{m}:{side}"):
            strat.setdefault(key, []).append(r)
        if m == "ml":
            b = "<1.4" if o < 1.4 else "1.4-1.7" if o < 1.7 else "1.7-2.1" if o < 2.1 else "2.1-2.7" if o < 2.7 else ">=2.7"
            strat.setdefault(f"ml:odds {b}", []).append(r)
    rep["strategies"] = {k: summarize(v) for k, v in sorted(strat.items())}
    rep["value"] = value_vs_intl(games)
    return rep


def value_vs_intl(games):
    """運彩大小分盤口 vs 國際盤等效盤口：運彩盤口比國際高 → 押小，反之押大；讓分同理。"""
    res = {}
    for g in games:
        s, diff = g["as"] + g["hs"], g["hs"] - g["as"]
        co, cu = g["tot"]
        I = intl_line(g["intl_tot"][0]) or intl_line(g["intl_tot"][1])
        T, oo, ou = (num(co["line"]) if co else None), (num(co["val"]) if co else None), (num(cu["val"]) if cu else None)
        if None not in (I, T, oo, ou):
            d = T - I
            if d != 0:
                o, won, push = (ou, s < T, s == T) if d > 0 else (oo, s > T, s == T)
                res.setdefault(f"tot gap {min(abs(d), 2):.2f}".replace("2.00", ">=2"), []).append(1.0 if push else o * won)
        ca, ch = g["hcp"]
        la = num(ca["line"]) if ca else None
        ia, ih = intl_line(g["intl_hcp"][0]), intl_line(g["intl_hcp"][1])
        if la is not None and (ia is not None or ih is not None) and ca["val"] and ch["val"]:
            # 國際盤口寫在讓分方：主隊那列有值＝主讓；換算成「客隊得到的讓分」
            intl_away = ih if ih is not None else -ia
            d = la - intl_away  # >0：運彩給客隊的分比國際多 → 押客
            if d != 0:
                m = -diff + la
                o, won = (num(ca["val"]), m > 0) if d > 0 else (num(ch["val"]), m < 0)
                res.setdefault(f"hcp gap {min(abs(d), 2):.2f}".replace("2.00", ">=2"), []).append(1.0 if m == 0 else o * won)
    return {k: summarize(v) for k, v in sorted(res.items())}


def main():
    leagues = [p.stem for p in sorted(DATA.glob("*.jsonl"))]
    report = {lg: r for lg in leagues if (r := league_report(lg))["years"]}
    OUT.mkdir(exist_ok=True)
    (OUT / "sports.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    ntests = sum(len(r["strategies"]) + len(r["value"]) for r in report.values())
    print(f"檢定數 {ntests}，Bonferroni 門檻 p < {0.05 / ntests:.2e}")
    for lg, r in report.items():
        print(f"\n== {lg}：{r['games']} 場，{r['years'][0]}–{r['years'][-1]}")
        for mk, yv in r["payout"].items():
            print(f"  {mk} 回收率 " + " ".join(f"{y}:{v:.3f}" for y, v in yv.items()))
        for title, block in (("固定押法", r["strategies"]), ("運彩 vs 國際盤價差", r["value"])):
            print(f"  -- {title}")
            for k, s in block.items():
                if s:
                    flag = " ★" if s["p_gt_breakeven"] < 0.05 / ntests else ""
                    print(f"    {k:18s} n={s['n']:6d} ROI={s['roi']:+.3f} ±{1.96 * s['se']:.3f} p={s['p_gt_breakeven']:.3f}{flag}")


if __name__ == "__main__":
    assert intl_line({"line": "9贏", "val": "50%"}) == 8.75 and intl_line({"line": "1分輸", "val": "50%"}) == 1.25
    assert intl_line({"line": "8.5", "val": ""}) == 8.5
    main()
