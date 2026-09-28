"""抓玩運彩（playsport.cc）「賽事結果查詢」頁：每場比賽的比分、運彩盤（讓分/不讓分/大小 的盤口與賠率）與國際盤口。
robots.txt 全站 Allow；每頁間隔 >= 1 秒、單執行緒。
用法：python src/sports/scrape.py MLB 2019-03-15 2026-09-27   → data/sports/MLB.jsonl（可中斷續抓）"""
import html
import json
import re
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

URL = "https://www.playsport.cc/gamesData/result?allianceid={aid}&gametime={d}"
LEAGUES = {"MLB": 1, "NPB": 2, "NBA": 3, "CPBL": 6, "KBO": 9}
OFF_MONTHS = {"NBA": {7, 8, 9}}  # 其餘聯盟 12–2 月休季
DEFAULT_OFF = {12, 1, 2}
OUT = Path(__file__).resolve().parents[2] / "data" / "sports"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36 (research; 1 req/s)"


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:  # 重試後仍失敗就往上丟
            if i == tries - 1:
                raise RuntimeError(f"{url}: {e}") from e
            time.sleep(5 * (i + 1))


def text(s):
    return html.unescape(re.sub(r"<[^>]+>", " ", s)).split()


def cell(row, cls):
    """回傳某欄的 (標記文字, 盤口, 賠率/百分比, 是否為贏方)。"""
    m = re.search(rf'<td class="{cls}">\s*<div([^>]*)>(.*?)</div>', row, re.S)
    if not m:
        return None
    win = "result-w" in m.group(1)
    strongs = [html.unescape(re.sub(r"<[^>]+>", "", x)).strip() for x in re.findall(r"<strong[^>]*>(.*?)</strong>", m.group(2), re.S)]
    spans = [html.unescape(re.sub(r"<[^>]+>", "", x)).strip(" ,") for x in re.findall(r"<span>(.*?)</span>", m.group(2), re.S)]
    side = strongs[0] if strongs else ""
    line = strongs[1] if len(strongs) > 1 else ""
    val = spans[-1] if spans else ""
    return {"side": side, "line": line, "val": val, "win": win}


def parse(page, day):
    """每場比賽是兩個帶 gameid 的 <tr>（上列＝客/大，下列＝主/小），依位置切段。"""
    marks = [(m.start(), m.group(1)) for m in re.finditer(r'<tr[^>]*gameid="(\d+)"', page)]
    by = {}
    for (pos, gid), nxt in zip(marks, marks[1:] + [(len(page), None)]):
        by.setdefault(gid, []).append(page[pos:nxt[0]])
    games = []
    for gid, segs in by.items():
        if len(segs) < 2:
            continue
        r1, r2 = segs[0], segs[1]
        lis = re.findall(r"<li[^>]*>\s*([^<]*?)\s*</li>", r1.split('class="td-universal-bet01"')[0])
        teams = re.findall(r'<td class="(?:secondteam|winnerteam)">\s*<a[^>]*>([^<]+)</a>', r1)
        if len(lis) < 3 or len(teams) < 2 or not lis[0].isdigit() or not lis[2].isdigit():
            continue  # 延賽/取消
        g = {"gid": gid, "date": day, "away": teams[0].strip(), "home": teams[1].strip(),
             "as": int(lis[0]), "hs": int(lis[2])}
        for key, cls in (("intl_hcp", "td-universal-bet01"), ("intl_tot", "td-universal-bet02"),
                         ("hcp", "td-bank-bet01"), ("ml", "td-bank-bet03"), ("tot", "td-bank-bet02")):
            g[key] = [cell(r1, cls), cell(r2, cls)]  # [客/大, 主/小]
        games.append(g)
    return games


def main(league, start, end):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{league}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["_day"] for l in path.read_text(encoding="utf-8").splitlines() if l.startswith('{"_day"')}
    d, end = date.fromisoformat(start), date.fromisoformat(end)
    with path.open("a", encoding="utf-8") as f:
        while d <= end:
            ds = d.strftime("%Y%m%d")
            if ds not in done and d.month not in OFF_MONTHS.get(league, DEFAULT_OFF):
                games = parse(get(URL.format(aid=LEAGUES[league], d=ds)), ds)
                f.write(json.dumps({"_day": ds, "n": len(games)}) + "\n")
                for g in games:
                    f.write(json.dumps(g, ensure_ascii=False) + "\n")
                f.flush()
                time.sleep(1.0)
            d += timedelta(days=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
