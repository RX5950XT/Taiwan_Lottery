# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "scikit-learn==1.9.1", "lightgbm==4.7.0"]
# ///
"""只看過去資料的彩券回測。執行：uv run src/lottery/ml_backtest.py

內建檢查：uv run src/lottery/ml_backtest.py --self-test
只寫 output/ml_results.{md,json}；不更動輸入與其他檔案。
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import binomtest, hypergeom
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
START, WARMUP, STEP, SEED = 500, 100, 50, 20260928
NAMES = {
    "random": "純隨機", "hot10": "熱號 10", "hot30": "熱號 30",
    "hot100": "熱號 100", "cold_gap": "冷號：最久未出",
    "cold100": "冷號：100 期最少", "logistic": "逐號 Logistic",
    "lightgbm": "逐號 LightGBM", "mlp": "10 期序列 MLP",
}
MODELS = ("logistic", "lightgbm", "mlp")


def load_game(game, size):
    path = ROOT / "data" / "lottery" / f"{game}.json"
    raw = path.read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) <= START:
        raise ValueError(f"{path}: 資料不足或不是陣列")
    periods, dates = [], []
    y = np.zeros((len(rows), size), dtype=np.int8)
    second = np.zeros((len(rows), 8), dtype=np.int8) if game == "power" else None
    for t, row in enumerate(rows):
        period, nums = row["period"], row["drawNumberSize"]
        if type(period) is not int or (periods and period <= periods[-1]):
            raise ValueError(f"{game} 第 {t} 筆：period 重複或未遞增")
        if (not isinstance(nums, list) or len(nums) != 7
                or any(type(x) is not int for x in nums)
                or nums[:6] != sorted(set(nums[:6]))
                or any(not 1 <= x <= size for x in nums[:6])
                or not 1 <= nums[6] <= (8 if second is not None else size)
                or (second is None and nums[6] in nums[:6])):
            raise ValueError(f"{game} 第 {t} 筆：號碼不合法")
        date = row["lotteryDate"][:10]
        if dates and date < dates[-1]:
            raise ValueError(f"{game} 日期未遞增")
        periods.append(period)
        dates.append(date)
        y[t, np.array(nums[:6]) - 1] = 1
        if second is not None:
            second[t, nums[6] - 1] = 1
    return y, second, {"periods": periods, "dates": dates,
                       "sha256": hashlib.sha256(raw).hexdigest()}


def features(y):
    """第 t 列只讀 y[:t]，最後一列供真正下一期使用。"""
    n, size = y.shape
    cumulative = np.vstack([np.zeros(size), np.cumsum(y, axis=0)])
    x = np.zeros((n + 1, size, 16), dtype=np.float64)
    sequence = np.zeros((n + 1, size * 10), dtype=np.float64)
    gaps = np.zeros((n + 1, size), dtype=np.float64)
    last = np.full(size, -1)
    for t in range(n + 1):
        gaps[t] = t - last - 1  # 未出現者以可觀測歷史長度計，不假裝知道 2008 以前
        for j, k in enumerate((5, 10, 30, 100)):
            x[t, :, j] = (cumulative[t] - cumulative[max(0, t - k)]) / max(1, min(k, t))
        x[t, :, 4] = np.log1p(gaps[t])
        x[t, :, 5] = cumulative[t] / max(1, t)
        for lag in range(1, 11):
            if t >= lag:
                x[t, :, 5 + lag] = y[t - lag]
                sequence[t, (lag - 1) * size:lag * size] = y[t - lag]
        if t < n:
            last[y[t].astype(bool)] = t
    return x, sequence, gaps


def train_predict(kind, x, sequence, y, stop, end):
    """訓練 label 僅 [100, stop)，預測 [stop, end)。不做測試集調參。"""
    train = slice(WARMUP, stop)
    test = slice(stop, end)
    if kind == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(32,), alpha=10.0, max_iter=100,
                              batch_size=min(128, stop - WARMUP), early_stopping=False,
                              random_state=SEED, tol=1e-4, n_iter_no_change=10)
        model.fit(sequence[train], y[train])
        return model.predict_proba(sequence[test])
    prediction = np.empty((end - stop, y.shape[1]))
    for number in range(y.shape[1]):
        target = y[train, number]
        if len(np.unique(target)) < 2:
            prediction[:, number] = (target.sum() + 1) / (len(target) + 2)
            continue
        if kind == "logistic":
            model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
        else:
            model = LGBMClassifier(n_estimators=60, learning_rate=0.03, num_leaves=7,
                                   max_depth=3, min_child_samples=50, reg_lambda=10,
                                   verbosity=-1, n_jobs=1, random_state=SEED,
                                   deterministic=True, force_col_wise=True)
        model.fit(x[train, number], target)
        # Booster 直接接 numpy，避免 sklearn 對 LightGBM 自動特徵名稱的無關警告。
        prediction[:, number] = (model.booster_.predict(x[test, number]) if kind == "lightgbm"
                                  else model.predict_proba(x[test, number])[:, 1])
    return prediction


def heuristic_probabilities(y, gaps):
    n, size = y.shape
    count = int(y[0].sum())
    prior = count / size
    cumulative = np.vstack([np.zeros(size), np.cumsum(y, axis=0)])
    result = {"random": np.full((n + 1, size), prior)}
    for window in (10, 30, 100):
        t = np.arange(n + 1)
        result[f"hot{window}"] = (
            cumulative - cumulative[np.maximum(0, t - window)] + 20 * prior
        ) / (np.minimum(t, window)[:, None] + 20)
    result["cold100"] = (1 - result["hot100"]) * count / (size - count)
    # 同樣遺漏期數給相同機率；越久未出者權重越大。
    weights = 1 + gaps / (gaps + 1 / prior)
    result["cold_gap"] = weights / weights.sum(axis=1, keepdims=True) * count
    return result


def tickets(probability, count, seed, random=False):
    rng = np.random.default_rng(seed)
    scores = rng.random(probability.shape) if random else probability
    # 用第二排序鍵打破完全相等的分數，不讓小號碼固定優先。
    order = np.lexsort((rng.random(probability.shape), scores), axis=1)
    return np.sort(order[:, -count:], axis=1)


def losses(y, p, prior):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ll = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    bs = (y - p) ** 2
    base_ll = -(y * math.log(prior) + (1 - y) * math.log1p(-prior))
    base_bs = (y - prior) ** 2
    per_number = [{"number": j + 1, "log_loss": float(ll[:, j].mean()),
                   "brier": float(bs[:, j].mean()),
                   "baseline_log_loss": float(base_ll[:, j].mean()),
                   "baseline_brier": float(base_bs[:, j].mean())} for j in range(y.shape[1])]
    return {"log_loss": float(ll.mean()), "brier": float(bs.mean()),
            "baseline_log_loss": float(base_ll.mean()), "baseline_brier": float(base_bs.mean()),
            "per_number": per_number}


def total_hits_null(size, draws):
    """總命中數是 draws 個 Hypergeom(size,6,6) 的和；不是 6*draws 次獨立 Bernoulli。"""
    pmf = hypergeom.pmf(np.arange(7), size, 6, 6)
    # 小型多項式反覆卷積，免去 FFT 尾端負數／精度處理，實際資料僅約兩千期。
    distribution = np.array([1.0])
    for _ in range(draws):
        distribution = np.convolve(distribution, pmf)
    return distribution / distribution.sum()


def run_game(game, size, warning_counts):
    y, second, metadata = load_game(game, size)
    domains = {"main": (y, 6)}
    if second is not None:
        domains["second"] = (second, 1)
    probabilities, next_probabilities, selected = {}, {}, {}
    for domain, (target, count) in domains.items():
        x, sequence, gaps = features(target)
        all_p = heuristic_probabilities(target, gaps)
        for kind in MODELS:
            all_p[kind] = np.full((len(y) + 1, target.shape[1]), np.nan)
        for stop in range(START, len(y), STEP):
            end = min(stop + STEP, len(y))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                for kind in MODELS:
                    all_p[kind][stop:end] = train_predict(kind, x, sequence, target, stop, end)
            warning_counts.update(f"{game}/{domain}: {w.category.__name__}: {w.message}" for w in caught)
            if (stop - START) % 250 == 0:
                print(f"{game}/{domain}: 已預測至第 {end}/{len(y)} 期", flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            for kind in MODELS:
                all_p[kind][-1:] = train_predict(kind, x, sequence, target, len(y), len(y) + 1)
        warning_counts.update(f"{game}/{domain}: {w.category.__name__}: {w.message}" for w in caught)
        probabilities[domain] = {key: value[START:-1] for key, value in all_p.items()}
        next_probabilities[domain] = {key: value[-1] for key, value in all_p.items()}
        selected[domain] = {}
        for key, value in all_p.items():
            if not np.isfinite(value[START:]).all() or not ((value[START:] >= 0) & (value[START:] <= 1)).all():
                raise ValueError(f"{game}/{domain}/{key}: 模型機率不合法")
            selected[domain][key] = tickets(value[START:], count, SEED + size + count, key == "random")
    return summarize(game, size, y, second, metadata, probabilities, next_probabilities, selected)


def summarize(game, size, y, second, metadata, probabilities, next_p, selected):
    observed = y[START:]
    n = len(observed)
    distribution = total_hits_null(size, n)
    tail3 = float(hypergeom.sf(2, size, 6, 6))
    results = {}
    for key in NAMES:
        ticket = selected["main"][key][:-1]
        hits = np.take_along_axis(observed, ticket, axis=1).sum(axis=1)
        row = {"mean_hits": float(hits.mean()), "rate_ge3": float((hits >= 3).mean()),
               "hit_histogram": np.bincount(hits, minlength=7).tolist(),
               "main_scores": losses(observed, probabilities["main"][key], 6 / size),
               "p_raw": {"mean_hits": float(distribution[int(hits.sum()):].sum()),
                         "rate_ge3": float(binomtest(int((hits >= 3).sum()), n, tail3,
                                                    alternative="greater").pvalue)},
               "next_ticket": {"main": (selected["main"][key][-1] + 1).tolist()},
               "next_probabilities": {"main": next_p["main"][key].tolist()},
               "audit": {"tickets": (ticket + 1).tolist(), "hits": hits.tolist(),
                         "probabilities_main": probabilities["main"][key].tolist()}}
        if second is not None:
            second_ticket = selected["second"][key][:-1]
            successes = np.take_along_axis(second[START:], second_ticket, axis=1)[:, 0]
            row["second_rate"] = float(successes.mean())
            row["second_scores"] = losses(second[START:], probabilities["second"][key], 1 / 8)
            row["p_raw"]["second_rate"] = float(binomtest(int(successes.sum()), n, 1 / 8,
                                                       alternative="greater").pvalue)
            row["next_ticket"]["second"] = int(selected["second"][key][-1, 0] + 1)
            row["next_probabilities"]["second"] = next_p["second"][key].tolist()
            row["audit"].update(second_tickets=(second_ticket[:, 0] + 1).tolist(),
                                 second_hits=successes.tolist(),
                                 probabilities_second=probabilities["second"][key].tolist())
        results[key] = row
    return {"name": "威力彩" if game == "power" else "大樂透", "size": size,
            "total_draws": len(y), "test_draws": n, "input": metadata,
            "test_periods": metadata["periods"][START:],
            "training_last_period": [metadata["periods"][START + ((t - START) // STEP) * STEP - 1]
                                     for t in range(START, len(y))],
            "theory": {"mean_hits": 36 / size, "rate_ge3": tail3,
                       "main_probability": 6 / size}, "strategies": results}


def correct_comparisons(games):
    count = sum(len(row["p_raw"]) for game in games.values() for row in game["strategies"].values())
    discoveries = []
    for key, game in games.items():
        for method, row in game["strategies"].items():
            row["p_bonferroni"] = {metric: min(1.0, p * count) for metric, p in row["p_raw"].items()}
            for metric, p in row["p_bonferroni"].items():
                if p < 0.05:
                    discoveries.append({"game": key, "method": method, "metric": metric, "p": p})
    return count, discoveries


def report(result):
    lines = ["# 彩券機器學習 walk-forward 回測", "", result["conclusion"], "",
             "## 方法與限制", "",
             "- 每種策略每期只買 1 注。前 500 期為初始歷史；模型使用第 101 期起的訓練樣本，前 100 期供特徵暖身。",
             "- 每 50 期從頭重訓，訓練範圍逐步增加。區塊內每一期特徵只讀前一期以前已開出的號碼；不是一次預測未來 50 期。",
             "- 每號碼各訓練一個 Logistic／LightGBM 二元模型；特徵是近 5/10/30/100 期頻率、log(遺漏期數+1)、歷史頻率、近 10 期逐期是否出現。",
             "- 逐號模型內奇偶、號碼區間是常數，故不加入無效欄位。MLP 以最近 10 期完整 one-hot 向量展平作有順序的輸入，32 個隱藏節點，多標籤 sigmoid 輸出；不是 LSTM。",
             "- Logistic C=0.1；LightGBM 60 棵樹、深度 3、7 葉、learning_rate=0.03、reg_lambda=10；MLP alpha=10、最多 100 epochs。標準化只用訓練資料。未使用測試資料選參數、早停或挑選模型。",
             "- 熱號以 (近 N 期次數 + 20×基準機率)/(N+20) 提供平滑機率；冷號 100 期反轉熱號機率並重新縮放；最久未出以 1+gap/(gap+1/prior) 正規化。這些是明確定義的啟發式機率，不代表已校準。",
             "- 主號挑機率前 6 名，第二區挑前 1 名；平手以固定種子隨機打破。純隨機票不放回均勻抽取，預測機率仍為理論常數，不把抽到的票誤當 0/1 信心。",
             "- 大樂透特別號不當作第七個主號，也不另外選特別號；威力彩第二區獨立建模。",
             "- 虛無假設是各期獨立、所有合法組合均勻。平均命中數用超幾何分布的逐期卷積做單尾精確檢定；≥3 命中及第二區用單尾精確二項檢定。主號同一期不獨立，不能當 6 次 Bernoulli。",
             f"- 跨兩種彩券、所有策略與三種命中指標，共 {result['comparisons']} 個檢定一起 Bonferroni 校正，門檻 0.05。適應過去資料的選票在虛無假設下仍有相同條件命中分布，故不需有洩漏風險的事後洗牌。",
             "- p-value 對比理論均勻隨機，而非只和某一次隨機種子比高低。Log-loss／Brier 越低越好，按號碼及期別平均；機率分數僅為描述性比較，不對分數差或逐號差宣稱統計顯著。",
             "- 未拒絕隨機假設不等於證明不可能預測。回測不能保證下一期效果；若看完本報告後再調參，必須用新增、未看過的期數確認。未計獎金、成本或投資報酬。", ""]
    for game in result["games"].values():
        data = game["input"]
        lines += [f"## {game['name']}", "",
                  f"資料 {data['dates'][0]}～{data['dates'][-1]}，共 {game['total_draws']} 期；回測 {game['test_draws']} 期，period {game['test_periods'][0]}～{game['test_periods'][-1]}。",
                  f"理論平均命中 {game['theory']['mean_hits']:.6f}；≥3 比例 {game['theory']['rate_ge3']:.4%}；威力彩第二區基準 12.5%。", "",
                  "| 策略 | 平均命中 | ≥3 比例 | 第二區 | 主號 Log-loss | 主號 Brier |",
                  "|---|---:|---:|---:|---:|---:|"]
        for method, row in game["strategies"].items():
            scores = row["main_scores"]
            second = f"{row['second_rate']:.3%}" if "second_rate" in row else "—"
            lines.append(f"| {NAMES[method]} | {row['mean_hits']:.6f} | {row['rate_ge3']:.3%} | {second} | {scores['log_loss']:.6f} | {scores['brier']:.6f} |")
        baseline = game["strategies"]["random"]["main_scores"]
        lines += [f"\n固定機率基準：Log-loss {baseline['baseline_log_loss']:.6f}，Brier {baseline['baseline_brier']:.6f}。", "",
                  "| 策略 | 平均命中 p / 校正 p | ≥3 p / 校正 p | 第二區 p / 校正 p |",
                  "|---|---:|---:|---:|"]
        for method, row in game["strategies"].items():
            cells = [f"{row['p_raw'][m]:.6g} / {row['p_bonferroni'][m]:.6g}" if m in row["p_raw"] else "—"
                     for m in ("mean_hits", "rate_ge3", "second_rate")]
            lines.append(f"| {NAMES[method]} | " + " | ".join(cells) + " |")
        lines += ["", "### 下一期示範票", "", "所有模型以完整已知歷史重訓。以下是各策略輸出，不是有優勢的投注推薦；不猜測下一個 period。", "",
                  "| 策略 | 主號 | 第二區 |", "|---|---|---|"]
        for method, row in game["strategies"].items():
            ticket = row["next_ticket"]
            lines.append(f"| {NAMES[method]} | {', '.join(map(str, ticket['main']))} | {ticket.get('second', '—')} |")
        for domain, label in (("main_scores", "主號"), ("second_scores", "第二區")):
            if domain not in game["strategies"]["random"]:
                continue
            lines += ["", f"### {label}逐號機率評分", "", "<details><summary>展開各策略、各號碼與常數基準的分數</summary>", "",
                      "| 策略 | 號碼 | Log-loss | 基準 Log-loss | Brier | 基準 Brier |",
                      "|---|---:|---:|---:|---:|---:|"]
            for method, row in game["strategies"].items():
                for score in row[domain]["per_number"]:
                    lines.append(f"| {NAMES[method]} | {score['number']} | {score['log_loss']:.6f} | {score['baseline_log_loss']:.6f} | {score['brier']:.6f} | {score['baseline_brier']:.6f} |")
            lines += ["", "</details>", ""]
    lines += ["## 重跑與驗證", "", "```powershell", "uv run src/lottery/ml_backtest.py --self-test", "uv run src/lottery/ml_backtest.py", "```", "",
              "套件版本由檔首鎖定，隨機種子固定；CPU 單執行緒。JSON 保留原始資料 SHA-256、每期票／機率／命中、訓練截止期與完整分數，可重新核算。不同平台浮點運算仍可能有微小差異。", "",
              f"內建檢查：{result['self_test']}。", "",
              "訓練警告（不隱藏未收斂）：", ""]
    lines += [f"- {key}：{count} 次" for key, count in result["warnings"].items()] or ["- 無。"]
    lines += ["", "參考：[SciPy 精確二項檢定](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)、[MLP 多標籤機率](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)、[LightGBM 決定性設定](https://lightgbm.readthedocs.io/en/latest/Parameters.html)。", ""]
    return "\n".join(lines)


def self_test():
    rng = np.random.default_rng(17)
    y = np.zeros((130, 38), dtype=np.int8)
    for row in y:
        row[rng.choice(38, 6, replace=False)] = 1
    x, seq, gaps = features(y)
    prefix = features(y[:110])
    for full, past in zip((x, seq, gaps), prefix):
        np.testing.assert_array_equal(full[:111], past)
    changed = y.copy()
    changed[110:] = np.roll(changed[110:], 1, axis=1)
    for a, b in zip(features(y), features(changed)):
        np.testing.assert_array_equal(a[:111], b[:111])
    for kind, p in heuristic_probabilities(y, gaps).items():
        np.testing.assert_allclose(p.sum(axis=1)[100:], 6)
        picked = tickets(p[100:], 6, 12, kind == "random")
        assert all(len(set(row)) == 6 for row in picked)
        np.testing.assert_array_equal(picked, tickets(p[100:], 6, 12, kind == "random"))
    q = hypergeom.pmf(np.arange(7), 38, 6, 6)
    np.testing.assert_allclose(total_hits_null(38, 2), np.convolve(q, q), atol=1e-14)
    score = losses(y, np.full(y.shape, 6 / 38), 6 / 38)
    assert abs(score["log_loss"] - score["baseline_log_loss"]) < 1e-14
    assert abs(score["brier"] - score["baseline_brier"]) < 1e-14
    # 實際模型的訓練標籤／標準化也不能讀預測當期與未來。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for kind in MODELS:
            a = train_predict(kind, x, seq, y, 110, 111)
            xx, ss, _ = features(changed)
            b = train_predict(kind, xx, ss, changed, 110, 111)
            np.testing.assert_array_equal(a, b)
    return "PASS：未來資料擾動／前綴一致性、三模型防洩漏、票合法且可重現、超幾何卷積、基準分數"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="只執行內建檢查，不寫報告")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        checked = self_test()
        print(checked, flush=True)
        if args.self_test:
            return
        warning_counts = Counter()
        games = {game: run_game(game, size, warning_counts) for game, size in (("power", 38), ("lotto", 49))}
    count, discoveries = correct_comparisons(games)
    conclusion = ("Bonferroni 校正後，沒有任何策略的命中指標顯著勝過均勻純隨機（α=0.05）；不能把排行榜的小幅領先當作可預測證據。"
                  if not discoveries else "有命中指標通過 Bonferroni 校正；這是歷史樣本中的證據，仍需新增未見期數確認，不能保證下一期。")
    result = {"seed": SEED, "initial_history": START, "feature_warmup": WARMUP,
              "retrain_every": STEP, "python": platform.python_version(),
              "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "lightgbm")},
              "comparisons": count, "discoveries": discoveries, "conclusion": conclusion,
              "self_test": checked, "warnings": dict(warning_counts), "games": games}
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    (out / "ml_results.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    (out / "ml_results.md").write_text(report(result), encoding="utf-8")
    print("\n" + conclusion)
    for game in games.values():
        print(f"\n{game['name']}：回測 {game['test_draws']} 期；理論平均命中 {game['theory']['mean_hits']:.6f}")
        for method, row in game["strategies"].items():
            second = f"，第二區 {row['second_rate']:.2%}" if "second_rate" in row else ""
            print(f"  {NAMES[method]}：平均 {row['mean_hits']:.6f}，≥3 {row['rate_ge3']:.2%}{second}，平均命中校正 p={row['p_bonferroni']['mean_hits']:.6g}")
    print(f"\n{count} 項校正檢定；訓練警告 {sum(warning_counts.values())} 次（詳見報告）。")
    print(f"輸出：{out / 'ml_results.md'}\n輸出：{out / 'ml_results.json'}")


if __name__ == "__main__":
    main()
