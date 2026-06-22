"""數學驗證：對 compute.py 的核心計算做單元檢查。跑：python3 verify_math.py
要是哪條 assert 掛了 = 算錯，立刻知道。"""
import math
import numpy as np
import pandas as pd
import compute as C

ok = []
def check(name, cond, got=None):
    ok.append(cond)
    print(("  ✓ " if cond else "  ✗ ") + name + ("" if cond else f"   got={got}"))

print("=== BSM gamma（對照解析值）===")
# S=100,K=100,T=1,r=0,q=0,σ=0.2 → d1=0.1, N'(0.1)=0.396953, γ=0.396953/(100*0.2*1)=0.0198476
g = C.bs_gamma(100, 100, 1, 0, 0, 0.2)
check("ATM gamma ≈ 0.019848", abs(g - 0.0198476) < 1e-5, g)
# gamma 對稱性與正值
check("gamma > 0", g > 0, g)
# 深價外 gamma 應較小
g_otm = C.bs_gamma(100, 130, 1, 0, 0, 0.2)
check("OTM gamma < ATM gamma", g_otm < g, (g_otm, g))

print("=== BSM 價格（對照解析值 + put-call parity）===")
c = C.bs_price(100, 100, 1, 0, 0, 0.2, 'C')
check("ATM call ≈ 7.9656", abs(c - 7.9656) < 1e-3, c)
p = C.bs_price(100, 100, 1, 0, 0, 0.2, 'P')
# parity: C - P = S e^-qT - K e^-rT = 100 - 100 = 0  → P = C
check("put-call parity (r=q=0): P==C", abs(p - c) < 1e-6, (p, c))
# 一般 parity: C - P = S e^{-qT} - K e^{-rT}
S,K,T,r,q,sig = 20000,20500,0.1,0.015,0.03,0.25
cc = C.bs_price(S,K,T,r,q,sig,'C'); pp = C.bs_price(S,K,T,r,q,sig,'P')
lhs = cc - pp; rhs = S*math.exp(-q*T) - K*math.exp(-r*T)
check("put-call parity (一般): C-P == Se^-qT - Ke^-rT", abs(lhs - rhs) < 1e-4, (lhs, rhs))

print("=== IV 反推 round-trip（價格→IV→價格）===")
for cp, Kx, true_iv in [('C', 21000, 0.22), ('P', 19000, 0.30), ('C', 20000, 0.18)]:
    price = C.bs_price(20000, Kx, 0.12, 0.015, 0.03, true_iv, cp)
    iv = C.implied_vol(price, 20000, Kx, 0.12, 0.015, 0.03, cp)
    check(f"IV round-trip {cp} K={Kx} σ={true_iv}", (not math.isnan(iv)) and abs(iv - true_iv) < 1e-3, iv)
# stale / 無解 應回 nan（settle ≤ 內含值）
iv_bad = C.implied_vol(0.5, 20000, 19000, 0.1, 0.015, 0.03, 'P')  # 內含=1000，price=0.5 → 無解
check("settle<內含值 → nan", math.isnan(iv_bad), iv_bad)

print("=== gamma 到期地板（壓 T→0 暴衝）===")
# 近到期 gamma 用地板後，應等於用 12 日 T 算的 gamma（而非更大的近 0 暴衝值）
iv0 = 0.25
g_real_near = C.bs_gamma(20000, 20000, 3/365, 0.015, 0.03, iv0)      # 3 日真 T
g_floor = C.bs_gamma(20000, 20000, C.GAMMA_TENOR_FLOOR_DAYS/365, 0.015, 0.03, iv0)
check("3日真 gamma > 12日地板 gamma（確認地板有壓低）", g_real_near > g_floor, (g_real_near, g_floor))

print("=== rolling percentile（聰明錢/散戶用）===")
s = pd.Series([10, 20, 30, 40, 50])
r5 = s.rolling(5, min_periods=1).rank(pct=True)
check("最大值 percentile == 1.0", abs(r5.iloc[-1] - 1.0) < 1e-9, r5.iloc[-1])
s2 = pd.Series([50, 40, 30, 20, 10])
r5b = s2.rolling(5, min_periods=1).rank(pct=True)
check("最小值 percentile == 0.2 (1/5)", abs(r5b.iloc[-1] - 0.2) < 1e-9, r5b.iloc[-1])

print("=== 聰明錢分數映射（pct→1..10）===")
# score = (pct_spot+pct_fut)/2，再 1+sc*9
def smart_score(p1, p2): return round((1 + ((p1+p2)/2)*9), 2)
check("pct=(1,1) → 10", smart_score(1, 1) == 10.0, smart_score(1,1))
check("pct=(0,0) → 1", smart_score(0, 0) == 1.0, smart_score(0,0))
check("pct=(0.5,0.5) → 5.5", smart_score(0.5, 0.5) == 5.5, smart_score(0.5,0.5))

print("=== 散戶熱度（3 分量平均×10）與 市場溫度（(聰明錢+散戶)/2）===")
heat = lambda a,b,c: round((a+b+c)/3*10, 2)
check("熱度 pct=(1,1,1) → 10", heat(1,1,1) == 10.0, heat(1,1,1))
temp = lambda sm,rh: round((sm+rh)/2*10)/10
check("溫度 (10,10)→10", temp(10,10) == 10.0, temp(10,10))
check("溫度 (4.8,9.8)→7.3", temp(4.8,9.8) == 7.3, temp(4.8,9.8))
check("溫度 (1,0)→0.5", temp(1,0) == 0.5, temp(1,0))

print("=== HV21 年化（log報酬 std ×√252）===")
px = pd.Series([100,101,99,102,100,103,101,104,102,105,103,106,104,107,105,108,106,109,107,110,108,111])
lr = np.log(px/px.shift(1))
hv = lr.rolling(21).std()*math.sqrt(252)
check("HV21 為正且合理(<3)", (hv.iloc[-1] > 0) and (hv.iloc[-1] < 3), hv.iloc[-1])

print("=== GEX 單位（每1點→百萬/點 顯示）===")
# gex_total（十億/點）×1000 = 百萬/點。0.024 十億 = 24 百萬
check("0.024 十億/點 ×1000 = 24 百萬/點", round(0.024*1000,1) == 24.0, 0.024*1000)

print()
print("結果：%d/%d 通過" % (sum(ok), len(ok)))
import sys; sys.exit(0 if all(ok) else 1)
