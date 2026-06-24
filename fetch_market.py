#!/usr/bin/env python3
"""
TWGEX fetch_market.py — 自包含市場資料抓取器（給 GitHub Actions 雲端每日跑）

只抓 compute.py 需要的 5 份公開資料，全部來自 TWSE / TAIFEX / FinMind 公開端點：
  1. taiex_daily.csv          ← TWSE FMTQIK（加權指數 + 成交量，按月，民國日期）
  2. inst_investors_total.csv ← FinMind TaiwanStockTotalInstitutionalInvestors（大盤三大法人）
  3. margin_total.csv         ← FinMind TaiwanStockTotalMarginPurchaseShortSale（融資融券）
  4. taifex_inst_futures.csv  ← FinMind TaiwanFuturesInstitutionalInvestors（TX/MTX 期貨法人）
  5. txo_daily/<yr>.csv        ← TAIFEX optDataDown（台指選擇權逐履約價，GEX 用）

⚠ 本檔【刻意】不 import taistock —— 交易策略/回測等 edge 絕不進入公開 repo。
   這裡只有「下載公開政府資料」的程式，可安全公開。

用法：
  python fetch_market.py                 # 增量：從既有 CSV 最後日期往前留 7 天重疊 → 抓到今天
  python fetch_market.py --days 60       # 沒有既有資料時，回看 60 天
  python fetch_market.py --start 2024-01-01 --end 2024-03-31   # 指定區間
  python fetch_market.py --data-dir /tmp/test --days 5         # 測試用，不動正式 data
  FINMIND_TOKEN=xxx python fetch_market.py   # 選配：提高 FinMind 速率上限（匿名也能跑）
"""
from __future__ import annotations
import argparse, csv, io, os, re, time
from datetime import date, timedelta
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

FINMIND_API = "https://api.finmindtrade.com/api/v4/data"
FMTQIK = "https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK"
FMTQIK_LEGACY = "https://www.twse.com.tw/exchangeReport/FMTQIK"
OPT_DAILY_DOWNLOAD = "https://www.taifex.com.tw/cht/3/optDataDown"

THROTTLE = 0.6          # FinMind / TAIFEX 之間的禮貌間隔
TWSE_THROTTLE = 3.0     # TWSE 限流較嚴
FUT_PRODUCTS = {"TX": "臺股期貨", "MTX": "小型臺指", "TMF": "微型臺指"}  # 微台 2022-07 上市，散戶方向用
OPT_DAILY_COLMAP = [("交易日期", "date"), ("到期月份", "expiry"), ("履約價", "strike"),
                    ("買賣權", "cp"), ("收盤價", "close"), ("成交量", "volume"),
                    ("結算價", "settle"), ("未沖銷契約數", "oi"),
                    ("交易時段", "session"), ("契約到期日", "expiry_date")]


# ───────────────────────── 共用工具 ─────────────────────────
def log(m): print(m, flush=True)
def sleep(s):
    if s and s > 0: time.sleep(s)

def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "zh-TW,zh;q=0.9"})
    retry = Retry(total=4, connect=4, read=4, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET", "POST"]), raise_on_status=False)
    ad = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    s.mount("https://", ad); s.mount("http://", ad)
    return s

def to_num(v):
    if v is None: return None
    s = str(v).replace(",", "").replace("$", "").strip()
    if s in ("", "--", "---", "-", "X", "N/A", "null"): return None
    try:
        return int(s) if re.fullmatch(r"-?\d+", s) else float(s)
    except ValueError:
        return None

def roc_to_ad(s):
    if s is None: return s
    s = str(s).strip().replace("年", "/").replace("月", "/").replace("日", "")
    parts = [p for p in re.split(r"[/\-\.]", s) if p != ""]
    if len(parts) == 3:
        y, m, d = parts
    elif len(parts) == 1 and len(s) in (6, 7):
        y, m, d = s[:-4], s[-4:-2], s[-2:]
    else:
        return s
    try:
        return f"{int(y) + 1911:04d}-{int(m):02d}-{int(d):02d}"
    except ValueError:
        return s

def month_starts(start_date, end_date):
    s = pd.Timestamp(start_date).replace(day=1); e = pd.Timestamp(end_date)
    out = []
    while s <= e:
        out.append(s.strftime("%Y%m%d")); s = s + pd.offsets.MonthBegin(1)
    return out

def save_csv(df, path, key_cols, sort_cols=None):
    """增量寫入：與既有檔合併、依 key_cols 去重（新優先）、排序、原子寫入、utf-8-sig。"""
    if df is None or len(df) == 0:
        return _count(path)
    df = df.copy()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        try:
            combined = pd.concat([pd.read_csv(path, encoding="utf-8-sig"), df], ignore_index=True)
        except Exception:
            combined = df
    else:
        combined = df
    for k in key_cols:
        if k in combined.columns:
            combined[k] = combined[k].astype(str)
    combined = combined.drop_duplicates(subset=[k for k in key_cols if k in combined.columns], keep="last")
    sc = [c for c in (sort_cols or key_cols) if c in combined.columns]
    if sc:
        combined = combined.sort_values(sc).reset_index(drop=True)
    tmp = path + ".tmp"
    combined.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)
    return len(combined)

def _count(path):
    if not os.path.exists(path): return 0
    try: return len(pd.read_csv(path, encoding="utf-8-sig"))
    except Exception: return 0

def last_date(path, col="date"):
    if not os.path.exists(path): return None
    try:
        s = pd.read_csv(path, encoding="utf-8-sig", usecols=[col])[col].astype(str)
        s = s[s.str.match(r"\d{4}-\d{2}-\d{2}")]
        return s.max() if len(s) else None
    except Exception:
        return None


# ───────────────────────── FinMind ─────────────────────────
def fm_fetch(session, dataset, start, end, token="", data_id=None):
    params = {"dataset": dataset, "start_date": start, "end_date": end}
    if data_id: params["data_id"] = data_id
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = session.get(FINMIND_API, params=params, headers=headers, timeout=60)
    try:
        j = r.json()
    except ValueError:
        raise RuntimeError(f"{dataset}: 非 JSON 回應 (HTTP {r.status_code})")
    msg = str(j.get("msg", ""))
    if j.get("status") not in (200, None) and "success" not in msg.lower():
        raise RuntimeError(f"{dataset}: status={j.get('status')} msg={msg}")
    sleep(THROTTLE)
    return pd.DataFrame(j.get("data", []) or [])


def fetch_inst_total(session, start, end, token=""):
    """大盤三大法人買賣超（原樣存：buy,date,name,sell）。"""
    df = fm_fetch(session, "TaiwanStockTotalInstitutionalInvestors", start, end, token)
    log(f"  · 三大法人現貨 {len(df)} 筆")
    return df


def fetch_margin_total(session, start, end, token=""):
    """大盤融資融券餘額（原樣存：含 TodayBalance,name,...）。"""
    df = fm_fetch(session, "TaiwanStockTotalMarginPurchaseShortSale", start, end, token)
    log(f"  · 融資融券 {len(df)} 筆")
    return df


def fetch_inst_futures(session, start, end, token=""):
    """三大法人期貨未平倉（TX/MTX）→ date/product/role/long_oi/short_oi/net_oi/...長表。"""
    frames = []
    for fid, pname in FUT_PRODUCTS.items():
        df = fm_fetch(session, "TaiwanFuturesInstitutionalInvestors", start, end, token, data_id=fid)
        if df is None or df.empty:
            continue
        g = lambda c: pd.to_numeric(df.get(c), errors="coerce")
        loi, soi = g("long_open_interest_balance_volume"), g("short_open_interest_balance_volume")
        lt, st = g("long_deal_volume"), g("short_deal_volume")
        frames.append(pd.DataFrame({
            "date": df["date"], "product": pname, "role": df["institutional_investors"],
            "long_oi": loi, "short_oi": soi, "net_oi": loi - soi,
            "long_trade": lt, "short_trade": st, "net_trade": lt - st,
        }))
        log(f"  · 期貨法人 {pname}({fid}) {len(df)} 筆")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ───────────────────────── TWSE 加權指數 ─────────────────────────
def _twse_month(session, ym, max_retry=5):
    backoff = 8
    for attempt in range(max_retry + 1):
        url = FMTQIK if attempt % 2 == 0 else FMTQIK_LEGACY
        try:
            return session.get(url, params={"date": ym, "response": "json"}, timeout=30).json()
        except ValueError:
            if attempt < max_retry:
                log(f"  · TWSE {ym} 疑似限流，退避 {backoff}s"); sleep(backoff); backoff = min(backoff * 2, 90)
            else:
                return None
        except Exception:
            if attempt < max_retry:
                sleep(backoff); backoff = min(backoff * 2, 90)
            else:
                return None
    return None


def fetch_taiex(session, start, end):
    """date, volume_shares, turnover, trades, taiex, change（民國轉西元）。"""
    rows = []
    for ym in month_starts(start, end):
        j = _twse_month(session, ym)
        if not j or j.get("stat") != "OK":
            sleep(TWSE_THROTTLE); continue
        for d in j.get("data", []) or []:
            rows.append({"date": roc_to_ad(d[0]), "volume_shares": to_num(d[1]),
                         "turnover": to_num(d[2]), "trades": to_num(d[3]),
                         "taiex": to_num(d[4]), "change": to_num(d[5])})
        sleep(TWSE_THROTTLE)
    df = pd.DataFrame(rows)
    if len(df):
        df = df[(df["date"] >= start) & (df["date"] <= end)].drop_duplicates(subset=["date"])
    log(f"  · 加權指數/成交量 {len(df)} 筆")
    return df


# ───────────────────────── TAIFEX 選擇權日行情 ─────────────────────────
def _decode(content):
    for enc in ("big5", "cp950", "utf-8-sig", "utf-8"):
        try: return content.decode(enc)
        except UnicodeDecodeError: continue
    return content.decode("big5", errors="ignore")

def _pick(header, key_zh):
    for i, h in enumerate(header):
        if key_zh in h.replace("　", ""): return i
    return None

def _g(row, idx, key):
    i = idx.get(key)
    return row[i].strip() if (i is not None and i < len(row)) else ""

def _opt_daily_parse(text):
    reader = list(csv.reader(io.StringIO(text)))
    if len(reader) < 2: return []
    header = [c.strip() for c in reader[0]]
    idx = {key: _pick(header, zh) for zh, key in OPT_DAILY_COLMAP}
    if idx.get("date") is None or idx.get("strike") is None: return []
    out = []
    for row in reader[1:]:
        if not _g(row, idx, "date"): continue
        if "一般" not in _g(row, idx, "session"): continue   # 只取日盤
        cp_raw = _g(row, idx, "cp")
        cp = "C" if "買權" in cp_raw else ("P" if "賣權" in cp_raw else "")
        if not cp: continue
        strike = to_num(_g(row, idx, "strike"))
        if strike is None: continue
        ed = _g(row, idx, "expiry_date")
        ed = f"{ed[:4]}-{ed[4:6]}-{ed[6:]}" if (len(ed) == 8 and ed.isdigit()) else ed.replace("/", "-")
        out.append({"date": _g(row, idx, "date").replace("/", "-"),
                    "expiry": _g(row, idx, "expiry"), "expiry_date": ed,
                    "strike": strike, "cp": cp, "settle": to_num(_g(row, idx, "settle")),
                    "oi": to_num(_g(row, idx, "oi")), "volume": to_num(_g(row, idx, "volume")),
                    "close": to_num(_g(row, idx, "close"))})
    return out

def _post_opt_daily(session, ms, me, max_retry=4):
    backoff = 6
    for attempt in range(max_retry + 1):
        try:
            r = session.post(OPT_DAILY_DOWNLOAD, data={
                "down_type": "1", "commodity_id": "TXO",
                "queryStartDate": ms.strftime("%Y/%m/%d"),
                "queryEndDate": me.strftime("%Y/%m/%d")}, timeout=60)
            text = _decode(r.content)
            if r.status_code == 200 and ("履約價" in text or len(text.strip()) < 5):
                return text
            if attempt < max_retry:
                sleep(backoff); backoff = min(backoff * 2, 60)
            else:
                return text
        except Exception as e:
            if attempt < max_retry:
                sleep(backoff); backoff = min(backoff * 2, 60)
            else:
                log(f"  · TXO日行情 {ms:%Y-%m} 放棄：{e!r}"); return ""
    return ""

def fetch_opt_daily(session, start, end):
    """逐月抓選擇權日行情；整月空回則逐工作日補。"""
    out = []
    ts_start, ts_end = pd.Timestamp(start), pd.Timestamp(end)
    cur = ts_start.replace(day=1)
    while cur <= ts_end:
        ms = max(cur, ts_start); me = min(cur + pd.offsets.MonthEnd(0), ts_end)
        rows = _opt_daily_parse(_post_opt_daily(session, ms, me))
        if not rows and me > ms:
            day = ms
            while day <= me:
                if day.weekday() < 5:
                    rows.extend(_opt_daily_parse(_post_opt_daily(session, day, day))); sleep(THROTTLE)
                day += pd.Timedelta(days=1)
        out.extend(rows)
        log(f"  · TXO日行情 {ms:%Y-%m} 取得 {len(rows)} 筆")
        sleep(THROTTLE)
        cur = (cur + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)).replace(day=1)
    df = pd.DataFrame(out)
    if len(df):
        df = df.drop_duplicates(subset=["date", "expiry", "strike", "cp"])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
    return df


# ───────────────────────── 主流程 ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description="TWGEX 市場資料抓取（自包含、僅公開資料）")
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "market"))
    ap.add_argument("--start", help="起始日 YYYY-MM-DD（覆寫自動增量）")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--days", type=int, default=45, help="無既有資料時的回看天數")
    ap.add_argument("--overlap", type=int, default=7, help="增量時與既有資料重疊天數（防漏補正）")
    a = ap.parse_args()

    mkt = a.data_dir
    end = a.end
    token = os.environ.get("FINMIND_TOKEN", "")
    s = make_session()
    log(f"FinMind {'（已帶 token）' if token else '（匿名模式）'}　輸出 → {mkt}")

    def win(fname):
        """各檔的抓取起點：既有最後日期 − overlap；沒有則 end − days。"""
        if a.start:
            return a.start
        ld = last_date(os.path.join(mkt, fname))
        if ld:
            return (pd.Timestamp(ld) - pd.Timedelta(days=a.overlap)).strftime("%Y-%m-%d")
        return (pd.Timestamp(end) - pd.Timedelta(days=a.days)).strftime("%Y-%m-%d")

    # 1) 加權指數 + 量
    st = win("taiex_daily.csv")
    log(f"\n【加權指數】{st} ~ {end}")
    n = save_csv(fetch_taiex(s, st, end), os.path.join(mkt, "taiex_daily.csv"), ["date"])
    log(f"  ✓ taiex_daily.csv（累計 {n}）")

    # 2) 三大法人現貨
    st = win("inst_investors_total.csv")
    log(f"\n【三大法人現貨】{st} ~ {end}")
    n = save_csv(fetch_inst_total(s, st, end, token), os.path.join(mkt, "inst_investors_total.csv"), ["date", "name"])
    log(f"  ✓ inst_investors_total.csv（累計 {n}）")

    # 3) 融資融券
    st = win("margin_total.csv")
    log(f"\n【融資融券】{st} ~ {end}")
    n = save_csv(fetch_margin_total(s, st, end, token), os.path.join(mkt, "margin_total.csv"), ["date", "name"])
    log(f"  ✓ margin_total.csv（累計 {n}）")

    # 4) 期貨法人
    st = win("taifex_inst_futures.csv")
    log(f"\n【期貨法人】{st} ~ {end}")
    n = save_csv(fetch_inst_futures(s, st, end, token), os.path.join(mkt, "taifex_inst_futures.csv"), ["date", "product", "role"])
    log(f"  ✓ taifex_inst_futures.csv（累計 {n}）")

    # 5) 選擇權日行情（GEX 用）→ 依年份存
    yr_files = sorted([f for f in os.listdir(os.path.join(mkt, "txo_daily"))
                       if f.endswith(".csv")]) if os.path.isdir(os.path.join(mkt, "txo_daily")) else []
    last_yr_file = yr_files[-1] if yr_files else None
    if a.start:
        st = a.start
    elif last_yr_file:
        ld = last_date(os.path.join(mkt, "txo_daily", last_yr_file))
        st = (pd.Timestamp(ld) - pd.Timedelta(days=a.overlap)).strftime("%Y-%m-%d") if ld \
             else (pd.Timestamp(end) - pd.Timedelta(days=a.days)).strftime("%Y-%m-%d")
    else:
        st = (pd.Timestamp(end) - pd.Timedelta(days=a.days)).strftime("%Y-%m-%d")
    log(f"\n【選擇權日行情 TXO】{st} ~ {end}")
    txo = fetch_opt_daily(s, st, end)
    if len(txo):
        txo["_yr"] = txo["date"].str[:4]
        for yr, g in txo.groupby("_yr"):
            g = g.drop(columns="_yr")
            n = save_csv(g, os.path.join(mkt, "txo_daily", f"{yr}.csv"),
                         ["date", "expiry", "strike", "cp"], sort_cols=["date", "expiry", "strike", "cp"])
            log(f"  ✓ txo_daily/{yr}.csv（累計 {n}，本次 {len(g)}）")
    else:
        log("  · 無新選擇權資料")

    log("\n全部完成。")


if __name__ == "__main__":
    main()
