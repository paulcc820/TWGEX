"""
TWGEX compute.py — 計算引擎（snapshot + backfill 序列）

自包含：讀 TWGEX/data/market/*.csv（已複製副本）→ 輸出 site/data.json（歷史序列，R3 schema）。
  • 結構 GEX(A)：逐履約價 BSM gamma×OI，dealer_sign OFF→pinning（OI gamma 重心）
  • 聰明錢方向：外資現貨+期貨 252日 rolling percentile 加權幾何平均 → 1空~10多
  • 大盤+量、台VIX(HV21 proxy=21日年化HV)
依 BLUEPRINT.md R1~R7。用法：python3 compute.py [--gex-start 2024-01-01]

⚠ GEX 符號 dealer_sign 待三指紋實證（meta.dealer_sign_verified=false）。
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.stats import norm
    from scipy.optimize import brentq
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

DATA_DIR = Path(__file__).resolve().parent / "data" / "market"      # 本地自包含副本
OUT_PATH = Path(__file__).resolve().parent / "site" / "data.json"

R, Q = 0.015, 0.03
TXO_MULT = 50
MINI_WEIGHT = 50 / 200
MICRO_WEIGHT = 10 / 200          # 微型臺指 NT$10/點 → 大台(NT$200/點)等值權重
WINDOW = 252
NEAR_EXPIRY_CUTOFF_DAYS = 2       # 只剔結算日（dte<2）的死合約
GAMMA_TENOR_FLOOR_DAYS = 12       # gamma 用的到期時間地板：壓掉 T→0 的 1/√T 暴衝（IV 仍用真實 T 解）
SERIES_START = "2019-01-01"     # 序列起點（控制 JSON 大小；三大法人期貨自 2018 起）
GEX_START = "2024-01-01"        # GEX 回算起點（txo_daily 有 2018+，先 bound 控制 runtime）
PROFILE_BIN = 500               # price profile bin 寬（點）
PROFILE_MARGIN_START = "2019-01-01"   # 融資地圖起點
PROFILE_FUTURES_START = "2018-01-01"  # 外資期貨地圖起點（期貨資料自 2018）
EPS = 1e-6


def _read(name):
    df = pd.read_csv(DATA_DIR / name)
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ── BSM / IV ──
def _npdf(x): return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
def _ncdf(x):
    return float(norm.cdf(x)) if _HAS_SCIPY else 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_price(S, K, T, r, q, sig, cp):
    if T <= 0 or sig <= 0:
        return max(0.0, (S - K) if cp == "C" else (K - S))
    d1 = (math.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if cp == "C":
        return S * math.exp(-q * T) * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * math.exp(-q * T) * _ncdf(-d1)

def bs_gamma(S, K, T, r, q, sig):
    if T <= 0 or sig <= 0: return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T))
    return math.exp(-q * T) * _npdf(d1) / (S * sig * math.sqrt(T))

def implied_vol(price, S, K, T, r, q, cp):
    intrinsic = max(0.0, (S - K) if cp == "C" else (K - S))
    if price <= intrinsic + 1.0 or price <= 1.0:
        return float("nan")
    f = lambda s: bs_price(S, K, T, r, q, s, cp) - price
    if _HAS_SCIPY:
        try:
            if f(0.01) * f(5.0) > 0: return float("nan")
            return brentq(f, 0.01, 5.0, xtol=1e-6, maxiter=200)
        except Exception:
            return float("nan")
    lo, hi = 0.01, 5.0
    if f(lo) * f(hi) > 0: return float("nan")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0: hi = mid
        else: lo = mid
    return mid


# ── GEX（單日，從預先 group 的 txo 子表）──
def derive_expiry(code):
    """expiry_date 欄常為空（2024 全空/2025 多空）→ 用 expiry 月碼推算。
    月選（6 位 YYYYMM）＝該月第三個週三；非月碼（週選等）回 NaT 跳過。"""
    s = str(code).strip()
    if len(s) == 6 and s.isdigit():
        y, m = int(s[:4]), int(s[4:6])
        if not (1 <= m <= 12):
            return pd.NaT
        first = pd.Timestamp(y, m, 1)
        first_wed = 1 + ((2 - first.weekday()) % 7)   # 週三=2
        return pd.Timestamp(y, m, first_wed + 14)      # 第三個週三
    return pd.NaT

def gex_for_date(date, S, gday):
    if gday is None or gday.empty:
        return None
    rows = []
    for r in gday.itertuples(index=False):
        if not r.oi or pd.isna(r.settle):
            continue
        exp = r.expiry_date
        if pd.isna(exp):
            exp = derive_expiry(r.expiry)
        if pd.isna(exp):
            continue
        dte = (exp - date).days
        if dte < NEAR_EXPIRY_CUTOFF_DAYS:
            continue
        K, cp = float(r.strike), str(r.cp).strip()
        if cp == "C" and K < S * 0.995: continue
        if cp == "P" and K > S * 1.005: continue
        iv = implied_vol(float(r.settle), S, K, dte / 365.0, R, Q, cp)
        if math.isnan(iv): continue
        Tg = max(dte, GAMMA_TENOR_FLOOR_DAYS) / 365.0   # gamma 用地板 T，壓 T→0 暴衝
        g = bs_gamma(S, K, Tg, R, Q, iv)
        rows.append((K, g * float(r.oi) * S * TXO_MULT, g * float(r.oi)))
    if not rows:
        return None
    total = sum(x[1] for x in rows) / 1e9
    wsum = sum(x[2] for x in rows)
    pin = sum(K * w for K, _, w in rows) / wsum if wsum > 0 else None
    return {"total": round(total, 3), "pinning": round(pin) if pin else None}


def load_txo_groups(start_year):
    groups = {}
    for y in range(start_year, 2027):
        fp = DATA_DIR / "txo_daily" / f"{y}.csv"
        if not fp.exists(): continue
        df = pd.read_csv(fp)
        df.columns = [c.lstrip("﻿").strip() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
        for d, g in df.groupby("date"):
            groups[d] = g
    return groups


# ── 聰明錢方向序列（pandas rolling rank）──
def _spot_net(inst, name):
    d = inst[inst["name"] == name]
    return pd.Series(((d["buy"] - d["sell"]) / 1e8).values, index=d["date"]).sort_index()

def _fut_net(fut, role):
    d = fut[fut["role"] == role]
    big = d[d["product"] == "臺股期貨"].set_index("date")["net_oi"]
    mini = d[d["product"] == "小型臺指"].set_index("date")["net_oi"] * MINI_WEIGHT
    return big.add(mini, fill_value=0).sort_index()

def smart_series_all(inst, fut, idx):
    """三組聰明錢方向（各為「現貨+期貨淨額」的 252 日分位 → 1空~10多）：
    外資 / 自營(self+避險；投信移入散戶) / 加總(外資+自營)。透明、可解釋。"""
    def pr(s):
        s = s[~s.index.duplicated()].sort_index()
        return s.rolling(WINDOW, min_periods=60).rank(pct=True)
    def score(spot, futs):
        ps = pr(spot).reindex(idx)
        pf = pr(futs).reindex(idx)
        sc = pd.concat([ps, pf], axis=1).mean(axis=1)   # 現貨/期貨分位平均（缺一用另一）
        return (1 + sc * 9).round(2)
    fs = _spot_net(inst, "Foreign_Investor")
    ts = _spot_net(inst, "Investment_Trust")
    ds = _spot_net(inst, "Dealer_self").add(_spot_net(inst, "Dealer_Hedging"), fill_value=0)  # 自營 self+避險
    ff = _fut_net(fut, "外資")
    tf = _fut_net(fut, "投信")
    dfu = _fut_net(fut, "自營商")
    return {
        "foreign":  score(fs, ff),
        "domestic": score(ds, dfu),                          # 自營 self+避險（投信移入散戶熱度）
        "total":    score(fs.add(ds, fill_value=0),          # 聰明錢加總＝外資＋自營
                          ff.add(dfu, fill_value=0)),
    }


def retail_heat_series(taiex, inst, margin, fut, idx):
    """散戶熱度 0冷~10燙：散戶現貨壓力 + 融資變化 + 量能 + 散戶期貨方向，252 日分位平均。
    散戶期貨＝−(三大法人期貨淨額)，期貨為零和市場故由法人倒推（近似散戶＋其他非法人）。
    ⚠ 校準：量能/壓力是『絕對量』會隨大盤長期變大→分位永遠頂高失去鑑別力。
    故先 detrend（除以 60 日均值）再取分位，量測『相對近期常態的高低』。"""
    def pr(s):
        s = s[~s.index.duplicated()].sort_index()
        return s.rolling(WINDOW, min_periods=60).rank(pct=True)
    def detrend(s):
        s = s[~s.index.duplicated()].sort_index()
        return s / s.rolling(60, min_periods=20).mean()
    tdates = pd.DatetimeIndex(taiex["date"])
    turnover = pd.Series(taiex["turnover"].astype(float).values, index=tdates)
    tot = inst[inst["name"] == "total"]
    tot_net = pd.Series((tot["buy"].astype(float) - tot["sell"].astype(float)).values,
                        index=tot["date"]).sort_index()
    retail_pressure = turnover - tot_net.reindex(turnover.index).abs()
    mm = margin[margin["name"] == "MarginPurchaseMoney"]
    margin_chg = pd.Series(pd.to_numeric(mm["TodayBalance"], errors="coerce").values, index=mm["date"]).sort_index().diff()
    # 散戶期貨方向（零和倒推）：散戶 = −(外資 + 投信 + 自營商) 期貨淨額
    # 註：投信「期貨」屬基金避險仍留法人側扣除；只有投信「現貨」歸散戶
    retail_fut = -(_fut_net(fut, "外資").add(_fut_net(fut, "投信"), fill_value=0)
                   .add(_fut_net(fut, "自營商"), fill_value=0))
    # 投信現貨淨買超：ETF/基金申贖≈散戶 FOMO → 第 5 軸
    trust_spot = _spot_net(inst, "Investment_Trust")
    comp = pd.concat([pr(detrend(retail_pressure)).reindex(idx), pr(margin_chg).reindex(idx),
                      pr(detrend(turnover)).reindex(idx), pr(retail_fut).reindex(idx),
                      pr(trust_spot).reindex(idx)],
                     axis=1).mean(axis=1)
    return (comp * 10).round(2)


def retail_dir_series(fut, idx):
    """散戶方向 1空~10多：小台+微台（市值權重）零和倒推 −(三大法人合計)。
    大台剔除（保證金高、機構主導，不像散戶）；小台/微台為散戶主導商品，方向才是散戶自己的判斷。"""
    def pr(s):
        s = s[~s.index.duplicated()].sort_index()
        return s.rolling(WINDOW, min_periods=60).rank(pct=True)
    def prod_net(p):
        d = fut[fut["product"] == p]
        return d.groupby("date")["net_oi"].sum() if not d.empty else pd.Series(dtype=float)
    mini = prod_net("小型臺指") * MINI_WEIGHT
    micro = prod_net("微型臺指") * MICRO_WEIGHT
    retail_net = -(mini.add(micro, fill_value=0))           # 散戶＝−法人（零和）
    return (1 + pr(retail_net).reindex(idx) * 9).round(2)


def _profile_by_price(val, taiex, start, how="sum"):
    """把逐日序列 val(date-indexed) 依當日 taiex 點位 bin 到 PROFILE_BIN 聚合。
    how='sum'＝累加（流量／各價位「分配」到的量，如 Δ融資、Δ淨部位，volume-profile 同理）；
    how='mean'＝取平均（水位）。傳回 [{level: bin中心, value: round(聚合)}]。"""
    if val is None or len(val) == 0:
        return []
    df = pd.DataFrame({"date": pd.DatetimeIndex(val.index),
                       "v": pd.to_numeric(pd.Series(val.values), errors="coerce")})
    df = df.dropna(subset=["v"])
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    tx = taiex[["date", "taiex"]].rename(columns={"taiex": "price"})
    df = df.merge(tx, on="date", how="inner")
    if df.empty:
        return []
    df["bin"] = (df["price"] // PROFILE_BIN) * PROFILE_BIN + PROFILE_BIN / 2
    g = df.groupby("bin")["v"]
    out = (g.sum() if how == "sum" else g.mean()).reset_index().sort_values("bin")
    return [{"level": int(r["bin"]), "value": round(float(r["v"]))} for _, r in out.iterrows()]


def foreign_net_series(fut):
    """外資期貨淨部位逐日 Series(date-indexed)：大台 ＋ 小台×50/200 ＋ 微台×10/200 的
    net_oi（多單−空單）。>0 淨多、<0 淨空。與 _fut_net / retail_dir_series 同口徑（含微型臺指）。"""
    fgn = fut[fut["role"] == "外資"].copy()
    fgn = fgn[fgn["date"] >= pd.Timestamp(PROFILE_FUTURES_START)]

    def _prod(p, col):
        d = fgn[fgn["product"] == p][["date", "net_oi"]].copy()
        return d.rename(columns={"net_oi": col})
    c = (_prod("臺股期貨", "big")
         .merge(_prod("小型臺指", "mini"), on="date", how="outer")
         .merge(_prod("微型臺指", "micro"), on="date", how="outer"))
    for col in ["big", "mini", "micro"]:                        # 防呆 + 缺資料補 0
        c[col] = pd.to_numeric(c[col], errors="coerce").fillna(0)
    net = c["big"] + c["mini"] * MINI_WEIGHT + c["micro"] * MICRO_WEIGHT
    return pd.Series(net.values, index=pd.DatetimeIndex(c["date"])).sort_index()


def margin_profile(margin, taiex):
    """融資地圖：每日融資餘額（億）的當日淨變化(Δ)，依當日 taiex 點位 bin 到 500 點「累加」。
    呈現槓桿資金在「各價位帶」的分配／進出（非市場總量）。value>0＝該價位帶融資淨增（散戶加碼槓桿）、
    <0＝淨減（去槓桿／斷頭）。傳回 [{level: bin中心, value: 淨Δ融資億}]。"""
    mm = margin[margin["name"] == "MarginPurchaseMoney"].copy()
    bal = pd.Series(pd.to_numeric(mm["TodayBalance"], errors="coerce").values / 1e8,
                    index=pd.DatetimeIndex(mm["date"]))
    bal = bal[~bal.index.duplicated()].sort_index()
    return _profile_by_price(bal.diff(), taiex, PROFILE_MARGIN_START, "sum")


def foreign_short_profile(fut, taiex):
    """外資期貨淨部位地圖（同融資地圖邏輯）：外資淨部位（口, 大台等值, 多−空）的當日淨變化(Δ)，
    依當日 taiex 點位 bin 到 500 點「累加」。value>0＝該價位帶外資淨加多（或回補空）、
    <0＝淨加空（或減多）。傳回 [{level: bin中心, value: 淨Δ部位口}]。"""
    return _profile_by_price(foreign_net_series(fut).diff(), taiex, PROFILE_FUTURES_START, "sum")


def divergence_calc(Dser, Hser):
    """恐慌背離＝散戶熱 − 聰明錢熱（線性差值）。
    乘積公式在 d=0（聰明錢中性）時退化為固定 50，故改線性差值。
    gauge 0~100：>50 散戶獨熱（分配頂背離），<50 聰明錢獨熱（吸籌底背離），≈50 同步。"""
    d = (Dser - 5.5) / 4.5
    h = (Hser - 5.0) / 5.0
    divergence = h - d
    gauge = ((divergence + 2) / 4 * 100).clip(0, 100)
    return divergence.round(3), gauge.round(1)


def build_series():
    taiex = _read("taiex_daily.csv")
    inst = _read("inst_investors_total.csv")
    fut = _read("taifex_inst_futures.csv")

    taiex = taiex[taiex["date"] >= pd.Timestamp(SERIES_START)].sort_values("date").reset_index(drop=True)
    idx = pd.DatetimeIndex(taiex["date"])
    dates = [d.strftime("%Y-%m-%d") for d in idx]

    px = taiex["taiex"].astype(float)
    turnover_bn = (taiex["turnover"].astype(float) / 1e8).round(1)        # 億元（turnover 原單位元/1e8=億；保留原語意）

    # HV21 proxy（21日年化 HV）
    logret = np.log(px / px.shift(1))
    hv21 = (logret.rolling(21).std() * math.sqrt(252)).round(4)

    sm = smart_series_all(inst, fut, idx)

    # GEX 序列
    gstart = pd.Timestamp(GEX_START)
    groups = load_txo_groups(gstart.year)
    gex_total, gex_pin = [], []
    n_gex = 0
    for i, d in enumerate(idx):
        if d < gstart:
            gex_total.append(None); gex_pin.append(None); continue
        res = None
        try:
            res = gex_for_date(d, float(px.iloc[i]), groups.get(d))
        except Exception:
            res = None
        if res:
            gex_total.append(res["total"]); gex_pin.append(res["pinning"]); n_gex += 1
        else:
            gex_total.append(None); gex_pin.append(None)

    def nn(series):  # None for NaN
        return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else (round(float(v), 4)) for v in series]

    gex_since = next((dates[i] for i, v in enumerate(gex_total) if v is not None), None)
    sm_f = nn(sm["foreign"])
    sm_since = next((dates[i] for i, v in enumerate(sm_f) if v is not None), None)

    # 散戶熱度 + 過熱/過冷計（恐慌計頁用）
    margin = _read("margin_total.csv")
    # 讀取完整 taiex（含 2018 之前，供 profile 用）
    taiex_full = _read("taiex_daily.csv")
    retail = retail_heat_series(taiex, inst, margin, fut, idx)
    retail_dir = retail_dir_series(fut, idx)
    mm = margin[margin["name"] == "MarginPurchaseMoney"]
    mbal = pd.Series(pd.to_numeric(mm["TodayBalance"], errors="coerce").values / 1e8,
                     index=pd.DatetimeIndex(mm["date"]))
    mbal = mbal[~mbal.index.duplicated()].sort_index().reindex(idx)
    margin_bal = mbal.round(1)          # 融資餘額（億）
    margin_chg = mbal.diff().round(2)   # 融資日變動（億）
    # 橫向量地圖（price profile）
    mprofile = margin_profile(margin, taiex_full)
    fprofile = foreign_short_profile(fut, taiex_full)
    fnet = foreign_net_series(fut).reindex(idx)
    foreign_net = nn(fnet)              # 逐日外資淨部位（口, 大台等值；>0 淨多、<0 淨空），供游標讀數

    div, gauge = divergence_calc(sm["foreign"], retail)
    rh, fg = nn(retail), nn(gauge)
    def _lastv(arr):
        for v in reversed(arr):
            if v is not None:
                return v
        return None
    D_last, H_last, g_last, hv_last = _lastv(sm_f), _lastv(rh), _lastv(fg), _lastv(nn(hv21))
    RD_last = _lastv(nn(retail_dir))
    MB_last, MC_last = _lastv(nn(margin_bal)), _lastv(nn(margin_chg))
    regime = "neutral"
    if D_last is not None and H_last is not None:
        dd, hh = (D_last - 5.5) / 4.5, (H_last - 5) / 5
        if dd > 0.2 and hh > 0.2: regime = "aligned_bullish"
        elif dd < -0.2 and hh < -0.2: regime = "aligned_bearish"
        elif hh > 0.2 and dd < -0.2: regime = "overheated"
        elif hh < -0.2 and dd > 0.2: regime = "undercooled"

    out = {
        "meta": {
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "data_as_of": dates[-1],
            "taiex_since": dates[0],
            "gex_since": gex_since,
            "smart_money_since": sm_since,
            "dealer_sign_verified": False,
            "gex_unit": "billion_ntd_per_point",
            "smart_money_unit": "percentile_1_to_10",
            "n": len(dates), "n_gex": n_gex,
            "today": {"regime": regime,
                      "fear_score": (None if g_last is None else round(g_last / 10, 1)),
                      "fear_gauge": g_last, "retail_heat": H_last,
                      "smart_foreign": D_last, "retail_dir": RD_last,
                      "margin_bal": MB_last, "margin_chg": MC_last, "hv21": hv_last},
        },
        "dates": dates,
        "taiex": [round(float(v), 2) for v in px],
        "taiex_turnover_bn": nn(turnover_bn),
        "gex_total": gex_total,
        "gex_pinning": gex_pin,
        "smart_money": sm_f,                 # 預設＝外資（聰明錢），供 snapshot
        "smart_foreign": sm_f,
        "smart_domestic": nn(sm["domestic"]),
        "smart_total": nn(sm["total"]),
        "retail_heat": rh,
        "retail_dir": nn(retail_dir),
        "margin_bal": nn(margin_bal),
        "margin_chg": nn(margin_chg),
        "fear_gauge": fg,
        "hv21_proxy": nn(hv21),
        "margin_profile": mprofile,
        "foreign_short_profile": fprofile,
        "foreign_net": foreign_net,
    }
    return out


def main():
    global GEX_START
    ap = argparse.ArgumentParser()
    ap.add_argument("--gex-start", default=GEX_START)
    a = ap.parse_args()
    GEX_START = a.gex_start
    out = build_series()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    m = out["meta"]
    print(f"wrote {OUT_PATH}")
    print(f"  dates: {m['n']} ({out['dates'][0]} → {out['dates'][-1]})")
    print(f"  GEX days: {m['n_gex']} (since {m['gex_since']})  smart since {m['smart_money_since']}")
    print(f"  last: taiex={out['taiex'][-1]} gex={out['gex_total'][-1]} smart={out['smart_money'][-1]} hv21={out['hv21_proxy'][-1]}")


if __name__ == "__main__":
    main()
