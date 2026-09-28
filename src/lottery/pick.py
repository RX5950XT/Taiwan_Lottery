"""產生「冷門」選號：中獎機率跟任何號碼一樣，但中了之後跟別人撞號分獎的機會較低。
用法：python src/lottery/pick.py power 5    （彩種 power|lotto，注數）
依據 output/analysis.json 的號碼熱門係數；先跑 src/lottery/analysis.py。"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N = {"power": 38, "lotto": 49}


def looks_patterned(c):
    """人類愛買的『圖案』：連號≥3、等差數列、全在同一個十位數。模型量不到，直接排除。"""
    d = [b - a for a, b in zip(c, c[1:])]
    three_in_row = any(a == b == 1 for a, b in zip(d, d[1:]))
    return three_in_row or len(set(d)) == 1 or len({x // 10 for x in c}) == 1


def pick(game, k, pool=4000, seed=None):
    coef = json.loads((ROOT / "output" / "analysis.json").read_text(encoding="utf-8"))[game]["popularity"]["coef"]
    rnd = random.Random(seed)
    cands = set()
    while len(cands) < pool:
        c = tuple(sorted(rnd.sample(range(1, N[game] + 1), 6)))
        if not looks_patterned(c):
            cands.add(c)
    # 取最冷門的前 5%，再從中隨機挑 → 不會每個人都算出同一組（否則又變熱門）
    top = sorted(cands, key=lambda c: sum(coef[i - 1] for i in c))[: pool // 20]
    out = rnd.sample(top, k)
    if game == "power":  # 第二區資料量不足以估偏好，隨機
        return [(c, rnd.randint(1, 8)) for c in out]
    return [(c, None) for c in out]


if __name__ == "__main__":
    assert looks_patterned((1, 2, 3, 20, 30, 40)) and looks_patterned((5, 10, 15, 20, 25, 30))
    assert looks_patterned((40, 41, 43, 44, 46, 48)) and not looks_patterned((3, 17, 25, 33, 41, 46))
    game = sys.argv[1] if len(sys.argv) > 1 else "lotto"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    if game not in N or not 1 <= k <= 100:
        sys.exit("用法：python src/lottery/pick.py power|lotto 注數(1-100)")
    for c, sp in pick(game, k):
        print(" ".join(f"{x:02d}" for x in c) + (f"  第二區 {sp:02d}" if sp else ""))
