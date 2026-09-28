"""威力彩第二區熱門度：中 3 個主號時，第二區有中/沒中人數比 → 各號被選比例（隨機=0.125）。執行：uv run --with numpy,scipy python src/lottery/second_zone.py"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

DATA = Path(__file__).resolve().parents[2] / "data" / "lottery"
d = json.loads((DATA / "power.json").read_text(encoding="utf-8"))
rows=[]
for r in d:
    s=r['drawNumberSize'][6]; a=r['super638SeventhAssign']['winnerCount']; b=r['super638NinthAssign']['winnerCount']
    c=r['super638FifthAssign']['winnerCount']; e=r['super638SixthAssign']['winnerCount']
    if b>0: rows.append((s,a+c,b+e,r['lotteryDate'][:10]))
s=np.array([x[0] for x in rows]); y=np.array([x[1] for x in rows]); n=np.array([x[2] for x in rows])
share=y/(y+n)
print(len(rows),'draws; overall share',share.mean().round(4),'(random=0.125)')
for k in range(1,9):
    m=s==k; print(k, m.sum(), share[m].mean().round(4), (share[m].mean()/0.125*100-100).round(1),'%')
print('ANOVA', stats.f_oneway(*[share[s==k] for k in range(1,9)]))
half=len(rows)//2
a1=[share[:half][s[:half]==k].mean() for k in range(1,9)]; a2=[share[half:][s[half:]==k].mean() for k in range(1,9)]
print('first half',np.round(a1,4));print('second half',np.round(a2,4));print('spearman',stats.spearmanr(a1,a2))
