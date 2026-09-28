"""威力彩 / 大樂透：隨機性檢定、彩民選號偏好（由各獎中獎人數反推）、每注期望值。

執行：uv run --with numpy,scipy,pandas,scikit-learn python src/lottery/analysis.py
輸出：output/analysis.json（給報告用）＋ stdout 摘要
"""
import json
from math import comb, exp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(0)
TAX_FREE, TAX_RATE = 5000, 0.204  # >5000 元：所得稅 20% + 印花稅 0.4%

# 各獎：中獎條件 (主號命中數, 第二區/特別號命中?) ；None=不看
GAMES = {
    "power": dict(name="威力彩", n=38, price=100, prefix="super638", special_n=8,
                  tiers={"Jackpot": (6, True), "Second": (6, False), "Third": (5, True),
                         "Fourth": (5, False), "Fifth": (4, True), "Sixth": (4, False),
                         "Seventh": (3, True), "Eighth": (2, True), "Ninth": (3, False),
                         "Normal": (1, True)}),
    "lotto": dict(name="大樂透", n=49, price=50, prefix="", special_n=None,
                  tiers={"jackpot": (6, None), "second": (5, True), "third": (5, False),
                         "fourth": (4, True), "fifth": (4, False), "sixth": (3, True),
                         "seventh": (2, True), "normal": (3, False)}),
}
# 用來推估彩民偏好的高人數、條件固定的獎項
POP_TIERS = {"power": ["Normal", "Eighth", "Seventh", "Ninth", "Sixth"],
             "lotto": ["normal", "seventh", "sixth", "fifth"]}


def tier_prob(g, main_hit, sp):
    """單注中某獎的精確機率。"""
    n = g["n"]
    p_main = comb(6, main_hit) * comb(n - 6, 6 - main_hit) / comb(n, 6)
    if g["special_n"]:  # 威力彩：第二區獨立 1/8
        return p_main * (1 / g["special_n"] if sp else 1 - 1 / g["special_n"])
    if sp is None:
        return p_main
    # 大樂透：特別號是剩下 43 顆之一，要落在自己沒中的 6-main_hit 個號碼裡
    miss = 6 - main_hit
    p_sp = miss / (n - 6)
    return p_main * (p_sp if sp else 1 - p_sp)


def load(game):
    g = GAMES[game]
    rows = json.loads((ROOT / "data" / "lottery" / f"{game}.json").read_text(encoding="utf-8"))
    recs = []
    for r in rows:
        rec = dict(period=r["period"], date=r["lotteryDate"][:10],
                   main=r["drawNumberSize"][:6], sp=r["drawNumberSize"][6],
                   order=r["drawNumberAppear"][:6], sales=r["sellAmount"])
        for t in g["tiers"]:
            a = r[f"{g['prefix']}{t}Assign" if g["prefix"] else f"{t}Assign"]
            rec[f"{t}_w"], rec[f"{t}_pp"] = a["winnerCount"], a["perPrize"]
            rec[f"{t}_pool"], rec[f"{t}_carry"] = a["prize"], a["lastPrize"]
        recs.append(rec)
    return pd.DataFrame(recs)


# ---------- 1. 隨機性檢定 ----------
def randomness(game, df):
    g, n, T = GAMES[game], GAMES[game]["n"], len(df)
    X = np.zeros((T, n), dtype=int)
    for i, m in enumerate(df["main"]):
        X[i, np.array(m) - 1] = 1
    freq = X.sum(0)
    exp_f = T * 6 / n
    # 每期 6 顆不放回，同期號碼負相關 → 變異數要乘 (n-6)/(n-1) 的修正
    chi = ((freq - exp_f) ** 2 / exp_f).sum() * (n - 1) / (n - 6)
    out = {"draws": T, "freq": freq.tolist(), "expected_freq": exp_f,
           "chi2_main": chi, "p_main": 1 - stats.chi2.cdf(chi, n - 1),
           "hot": (np.argsort(-freq)[:6] + 1).tolist(), "cold": (np.argsort(freq)[:6] + 1).tolist()}
    # 分段（每約 5 年）看有沒有某段期間偏差
    out["blocks"] = []
    for blk in np.array_split(np.arange(T), 4):
        f = X[blk].sum(0)
        e = len(blk) * 6 / n
        c = ((f - e) ** 2 / e).sum() * (n - 1) / (n - 6)
        out["blocks"].append({"from": df["date"].iloc[blk[0]], "to": df["date"].iloc[blk[-1]],
                              "p": float(1 - stats.chi2.cdf(c, n - 1))})
    if g["special_n"]:
        sf = np.bincount(df["sp"], minlength=9)[1:]
        out["sp_freq"] = sf.tolist()
        out["p_sp"] = stats.chisquare(sf).pvalue
    else:
        sf = np.bincount(df["sp"], minlength=n + 1)[1:]
        out["sp_freq"] = sf.tolist()
        out["p_sp"] = stats.chisquare(sf).pvalue
    # 連續兩期重複號碼數 vs 超幾何分布
    overlap = (X[1:] & X[:-1]).sum(1)
    hg = stats.hypergeom(n, 6, 6)
    obs = np.bincount(overlap, minlength=7)[:7]
    expc = np.array([hg.pmf(k) for k in range(7)]) * (T - 1)
    keep = expc >= 5
    o2, e2 = np.append(obs[keep], obs[~keep].sum()), np.append(expc[keep], expc[~keep].sum())
    out["repeat_from_last"] = {"obs": obs.tolist(), "exp": expc.round(1).tolist(),
                               "p": stats.chisquare(o2, e2 * o2.sum() / e2.sum()).pvalue}
    # 形態：和值、連號、奇數個數 vs 蒙地卡羅
    sim = np.sort(np.argsort(RNG.random((200000, n)), 1)[:, :6] + 1, 1)
    def shape(a):
        a = np.asarray(a)
        return a.sum(1), (np.diff(a, axis=1) == 1).any(1), (a % 2).sum(1)
    s_o, c_o, odd_o = shape(np.stack(df["main"]))
    s_s, c_s, odd_s = shape(sim)
    out["sum_ks_p"] = stats.ks_2samp(s_o, s_s).pvalue
    out["consec_rate"] = [c_o.mean(), c_s.mean()]
    out["consec_p"] = stats.binomtest(int(c_o.sum()), T, c_s.mean()).pvalue
    odd_obs = np.bincount(odd_o, minlength=7)
    odd_exp = np.bincount(odd_s, minlength=7) / len(sim) * T
    out["odd_p"] = stats.chisquare(odd_obs, odd_exp).pvalue
    # 號碼 × 前一期是否出現 → 條件出現率（熱號是否延續）
    prev = X[:-1].astype(bool)
    out["p_appear_given_prev"] = [X[1:][prev].mean(), X[1:][~prev].mean(), 6 / n]
    return out


# ---------- 2. 彩民選號偏好 ----------
def features(main, n):
    x = np.zeros(n)
    x[np.array(main) - 1] = 1
    return x


def popularity(game, df):
    """每期各獎『實際中獎人數 / 隨機購買下的期望人數』= 這組開獎號碼有多熱門。
    對 log(比值) 做 ridge 迴歸（特徵=每個號碼是否開出）→ 得到每個號碼的熱門係數。"""
    g = GAMES[game]
    N = df["sales"] / g["price"]
    ys = []
    for t in POP_TIERS[game]:
        hit, sp = g["tiers"][t]
        e = N * tier_prob(g, hit, sp)
        ok = df[f"{t}_w"] > 0
        y = np.where(ok, np.log(df[f"{t}_w"].clip(lower=1) / e), np.nan)
        ys.append(y - np.nanmean(y))  # 扣掉整體尺度（銷售額口徑差、電腦選號比例）
    Y = np.nanmean(np.vstack(ys), 0)
    X = np.stack([features(m, g["n"]) for m in df["main"]])
    ok = ~np.isnan(Y)
    X, Y, dfv = X[ok], Y[ok], df[ok].reset_index(drop=True)
    # 時間切半：前半訓練、後半驗證 → 偏好是穩定可預測的（對比：號碼本身不可預測）
    half = len(Y) // 2
    m1 = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(X[:half], Y[:half])
    r2_oos = m1.score(X[half:], Y[half:])
    m = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(X, Y)
    coef = m.coef_
    # 形態特徵的影響（單獨迴歸，好解釋）
    arr = np.stack(dfv["main"])
    feat = pd.DataFrame({
        "n_le31": (arr <= 31).sum(1), "n_le12": (arr <= 12).sum(1),
        "consec_pairs": (np.diff(arr, axis=1) == 1).sum(1),
        "same_decade_max": [max(np.bincount(np.array(a) // 10)) for a in arr],
        "n_mult5": (arr % 5 == 0).sum(1), "has_7": (arr % 10 == 7).sum(1),
        "n_4": (arr % 10 == 4).sum(1),
    })
    F = (feat - feat.mean()) / feat.std()
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(F)), F.values], Y, rcond=None)
    shape_eff = dict(zip(feat.columns, beta[1:].round(4)))
    # 驗證：用 3 中獎訓練出的偏好去預測『命中 5 顆』的人數（完全沒拿來訓練）
    t5 = "Fourth" if game == "power" else "third"
    hit, sp = g["tiers"][t5]
    e5 = (dfv["sales"] / g["price"]) * tier_prob(g, hit, sp)
    score = X @ coef
    rho5 = stats.spearmanr(score, dfv[f"{t5}_w"] / e5)
    # 頭獎：分數最熱門 1/4 的期數 vs 最冷 1/4，實際頭獎中獎人數/期望
    jt = "Jackpot" if game == "power" else "jackpot"
    pj = tier_prob(g, *g["tiers"][jt])
    lam = (dfv["sales"] / g["price"]) * pj
    q = pd.qcut(score, 4, labels=False)
    jack = [{"quartile": int(k), "winners": int(dfv[f"{jt}_w"][q == k].sum()),
             "expected_if_random": float(lam[q == k].sum())} for k in range(4)]
    # 頭獎人數 ~ Poisson(λ·exp(a + b·score))：把「熱門分數」換算成「同號對手倍數」f
    w = dfv[f"{jt}_w"].values
    nll = lambda ab: -(w * (ab[0] + ab[1] * score) - lam * np.exp(ab[0] + ab[1] * score)).sum()
    a, b = __import__("scipy.optimize", fromlist=["minimize"]).minimize(nll, [0.0, 1.0]).x
    ranking = np.argsort(coef) + 1
    sims = np.argsort(RNG.random((100000, g["n"])), 1)[:, :6]
    f_rand = np.exp(a + b * coef[sims].sum(1))
    f_cold = float(np.exp(a + b * np.sort(coef)[:6].sum()))
    f_hot = float(np.exp(a + b * np.sort(coef)[-6:].sum()))
    return {"coef": coef.round(4).tolist(), "r2_oos": r2_oos, "alpha": m.alpha_,
            "poisson_ab": [a, b], "f_cold": f_cold, "f_hot": f_hot,
            "f_random_mean": float(f_rand.mean()), "f_random_p10_p90": np.percentile(f_rand, [10, 90]).tolist(),
            "coldest_combo": sorted((np.argsort(coef)[:6] + 1).tolist()),
            "hottest_combo": sorted((np.argsort(coef)[-6:] + 1).tolist()),
            "least_popular": ranking[:10].tolist(), "most_popular": ranking[::-1][:10].tolist(),
            "shape_effect": shape_eff, "validate_5hit_spearman": [rho5.statistic, rho5.pvalue],
            "jackpot_by_popularity_quartile": jack,
            "overall_ratio": {t: float(np.nanmedian(df[f"{t}_w"] / (N * tier_prob(g, *g["tiers"][t]))))
                              for t in POP_TIERS[game]}}


# ---------- 3. 每注期望值 ----------
def after_tax(x):
    return np.where(x > TAX_FREE, x * (1 - TAX_RATE), x)


def guarantee(game, dates):
    """頭獎保底（中信公告）：大樂透 1 億；威力彩 2008 年 2.5 億、2009-03-02 起 4 億、2010-09-09 起 2 億。"""
    if game == "lotto":
        return np.full(len(dates), 1e8)
    d = np.asarray(dates)
    return np.where(d >= "2010-09-09", 2e8, np.where(d >= "2009-03-02", 4e8, 2.5e8))


def ev(game, df, pop_factor=1.0, skip_jackpot=False):
    """每期每注的期望回收（元）。頭獎/貳獎是『池子被 K+1 人平分』，K~Poisson(λ)：
    E[1/(1+K)] = (1-e^-λ)/λ。pop_factor<1 代表你選的組合比平均冷門，同號對手較少。"""
    g = GAMES[game]
    N = df["sales"] / g["price"]
    tot, tot_tax = np.zeros(len(df)), np.zeros(len(df))
    shared = ["Jackpot", "Second"] if game == "power" else ["jackpot", "second"]
    for t, (hit, sp) in g["tiers"].items():
        p = tier_prob(g, hit, sp)
        if skip_jackpot and t == shared[0]:
            continue
        if t in shared:
            pool = (df[f"{t}_pool"] + df[f"{t}_carry"]).astype(float)
            if t in ("Jackpot", "jackpot"):  # 有人中時頭獎保底
                pool = np.maximum(pool, guarantee(game, df["date"]))
            lam = (N * p * pop_factor).clip(lower=1e-12)
            share = (1 - np.exp(-lam)) / lam
            val = pool * share
        else:
            val = df[f"{t}_pp"].astype(float)
            if game == "lotto" and t in ("third", "fourth"):  # 浮動獎：冷門組合分到較多
                val = val / pop_factor ** 0.5
        tot += p * val
        tot_tax += p * after_tax(val)
    return tot / g["price"], tot_tax / g["price"]


def ev_summary(game, df, f_cold):
    g = GAMES[game]
    r, rt = ev(game, df)
    _, rt_cold = ev(game, df, pop_factor=f_cold)
    jt = "Jackpot" if game == "power" else "jackpot"
    jp = df[f"{jt}_pool"] + df[f"{jt}_carry"]
    i = int(np.argmax(rt))
    # 頭獎池多大時，稅後期望值才會 ≥ 1（固定其他條件用近 2 年中位數）
    recent = df.tail(200)
    base_other = float(np.median(ev(game, recent, skip_jackpot=True)[1]))
    pj = tier_prob(g, *g["tiers"][jt])
    breakeven = {}
    for sales_mult in (1, 3, 6):
        Nn = float(recent["sales"].median()) / g["price"] * sales_mult
        lam = Nn * pj
        share = (1 - exp(-lam)) / lam
        need = (1 - base_other) * g["price"] / (pj * share * (1 - TAX_RATE))
        breakeven[f"sales_x{sales_mult}"] = {"tickets": Nn, "jackpot_needed": need}
    return {"mean_return": float(r.mean()), "mean_return_after_tax": float(rt.mean()),
            "median_after_tax": float(np.median(rt)),
            "max_after_tax": float(rt[i]), "max_date": df["date"].iloc[i], "max_jackpot": float(jp.iloc[i]),
            "draws_ev_ge_1_after_tax": int((rt >= 1).sum()),
            "cold_combo_mean_after_tax": float(rt_cold.mean()),
            "cold_combo_max_after_tax": float(rt_cold.max()),
            "draws_ev_ge_1_cold": int((rt_cold >= 1).sum()),
            "other_prizes_return_after_tax": base_other,
            "breakeven": breakeven,
            "series": {"date": df["date"].tolist(), "ev_after_tax": rt.round(4).tolist(),
                       "jackpot": jp.astype(float).tolist(), "sales": df["sales"].astype(float).tolist()},
            "prob": {t: tier_prob(g, *c) for t, c in g["tiers"].items()},
            "jackpot_hits": int((df[f"{jt}_w"] > 0).sum()),
            "jackpot_multi_winner_draws": int((df[f"{jt}_w"] > 1).sum()),
            "jackpot_top": df.assign(jp=jp).nlargest(5, "jp")[["date", "jp", f"{jt}_w"]].values.tolist()}


def main():
    result = {}
    for game in GAMES:
        df = load(game)
        result[game] = {"randomness": randomness(game, df), "popularity": popularity(game, df),
                        "ev": None}
        result[game]["ev"] = ev_summary(game, df, result[game]["popularity"]["f_cold"])
        R, P, E = (result[game][k] for k in ("randomness", "popularity", "ev"))
        print(f"\n=== {GAMES[game]['name']}  {R['draws']} 期 ===")
        print(f"主號頻率卡方 p={R['p_main']:.3f}；分段 p={[round(b['p'], 3) for b in R['blocks']]}；"
              f"第二區/特別號 p={R['p_sp']:.3f}")
        print(f"連兩期重複 p={R['repeat_from_last']['p']:.3f}；和值 KS p={R['sum_ks_p']:.3f}；"
              f"連號 p={R['consec_p']:.3f}；奇數個數 p={R['odd_p']:.3f}")
        print(f"上期開出→本期出現率 {R['p_appear_given_prev'][0]:.4f}，沒開出 {R['p_appear_given_prev'][1]:.4f}，"
              f"理論 {R['p_appear_given_prev'][2]:.4f}")
        print(f"偏好模型 out-of-sample R²={P['r2_oos']:.3f}；驗證(5 顆命中人數) spearman={P['validate_5hit_spearman']}")
        print(f"最冷門號碼 {P['least_popular']}；最熱門 {P['most_popular']}")
        print(f"形態效果 {P['shape_effect']}")
        print(f"同號對手倍數 f：最冷組合 {P['coldest_combo']} {P['f_cold']:.2f}、最熱 {P['hottest_combo']} {P['f_hot']:.2f}、隨機平均 {P['f_random_mean']:.2f} p10-90 {P['f_random_p10_p90']}")
        print(f"頭獎依熱門度四分位 {P['jackpot_by_popularity_quartile']}")
        print(f"每 1 元期望回收：稅前 {E['mean_return']:.3f}、稅後 {E['mean_return_after_tax']:.3f}；"
              f"最高 {E['max_after_tax']:.3f}（{E['max_date']}，頭獎池 {E['max_jackpot']/1e8:.2f} 億）；"
              f"稅後≥1 的期數 {E['draws_ev_ge_1_after_tax']}；冷門組合≥1 期數 {E['draws_ev_ge_1_cold']}")
        print(f"損益兩平頭獎池 {({k: round(v['jackpot_needed']/1e8, 1) for k, v in E['breakeven'].items()})} 億")
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output" / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, default=float),
                                                   encoding="utf-8")


if __name__ == "__main__":
    # 自我檢查：所有獎項機率加總 ≤ 1，且頭獎機率正確
    for gk, gg in GAMES.items():
        assert sum(tier_prob(gg, *c) for c in gg["tiers"].values()) < 1
    assert abs(1 / tier_prob(GAMES["power"], 6, True) - 22085448) < 1
    assert abs(1 / tier_prob(GAMES["lotto"], 6, None) - 13983816) < 1
    main()
