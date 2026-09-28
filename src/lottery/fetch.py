"""抓台灣彩券官方 API 的歷史開獎資料（威力彩、大樂透、今彩539、39/49樂合彩、3/4星彩）（2008-01 起），存成 data/lottery/*.json。"""
import json
import time
import urllib.request
from datetime import date
from pathlib import Path

API = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/{ep}?period&month={m}&pageNum=1&pageSize=50"
GAMES = {"power": ("SuperLotto638Result", "superLotto638Res"),
         "lotto": ("Lotto649Result", "lotto649Res"),
         "d539": ("Daily539Result", "daily539Res"),
         "m539": ("39M5Result", "m539Res"),
         "m649": ("49M6Result", "m649Res"),
         "d3": ("3DResult", "lotto3DRes"),
         "d4": ("4DResult", "lotto4DRes")}
OUT = Path(__file__).resolve().parents[2] / "data" / "lottery"


def months(start=(2008, 1)):
    y, m = start
    today = date.today()
    while (y, m) <= (today.year, today.month):
        yield f"{y}-{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                body = json.load(r)
            if body.get("rtCode") != 0:
                raise ValueError(f"rtCode={body.get('rtCode')} {body.get('rtMsg')}")
            return body["content"]
        except Exception as e:  # 重試後仍失敗就往上丟，不吞
            if i == tries - 1:
                raise RuntimeError(f"{url}: {e}") from e
            time.sleep(2 ** i)


def validate(game, row):
    if game not in ("power", "lotto"):
        assert row["sellAmount"] >= 0 and row["totalAmount"] >= 0, row["period"]
        return
    nums = row["drawNumberSize"]
    hi, special_hi = (38, 8) if game == "power" else (49, 49)
    main, sp = nums[:6], nums[6]
    assert len(nums) == 7 and len(set(main)) == 6 and all(1 <= n <= hi for n in main), row["period"]
    assert 1 <= sp <= special_hi, row["period"]
    if game == "lotto":
        assert sp not in main, row["period"]


def main(only=None):
    OUT.mkdir(exist_ok=True)
    for game, (ep, key) in GAMES.items():
        if only and game not in only:
            continue
        rows = []
        for m in months():
            content = get(API.format(ep=ep, m=m))
            rows += content[key]
            time.sleep(0.15)
        rows = sorted({r["period"]: r for r in rows}.values(), key=lambda r: r["period"])
        for r in rows:
            validate(game, r)
        (OUT / f"{game}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(game, len(rows), rows[0]["lotteryDate"][:10], "~", rows[-1]["lotteryDate"][:10])


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])  # 例：python src/lottery/fetch.py d539 d3；不給參數＝全部
