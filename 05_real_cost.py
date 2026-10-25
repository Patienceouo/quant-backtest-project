# -*- coding: utf-8 -*-
"""
05_real_cost.py
第 5 课：拆穿「止损越小越赚」的两个幻觉。

上一课扫止损 1%~20%，回测说「越小越赚」（1% 止损赚 1.3 万倍）。
但那是两个 bug 叠出来的假象：

  幻觉 1 = 未来函数（收益归因错位）
     老代码：strat = pos * ret
     意思是「今天收盘才决定的仓位，却吃到了今天一整天的涨跌」。
     买入日白拿当天涨幅，止损日白躲当天跌幅——等于偷看未来。
     修正：strat = pos.shift(1) * ret（昨天的仓位，吃今天的涨跌）

  幻觉 2 = 零成本（真实成本按「次数」放大）
     老代码成本只有 0.1%，没有佣金、印花税、滑点。
     真实：佣金(万2.5双边) + 印花税(卖出单边) + 滑点。
     止损越紧 → 交易次数越多 → 成本越高。

今天只证一件事：紧止损有真实的、按次数放大的成本，
所以「越小越赚」是模型幻觉。最优止损到底在哪，是下一步三段切分的事。

数据：上证指数（新浪源），只看近十年 2016-2026
（全程 1990-2026 成本历史失真太严重：90 年代佣金 0.3%~0.5%、印花税 0.3%~0.6%，
 不能拿一个数扫全程）
"""
import os
import numpy as np
import pandas as pd
import akshare as ak
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 拉数据（新浪源） ----------------
df = ak.stock_zh_index_daily(symbol="sh000001")
df = df[["date", "close"]].copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------------- 真实成本（可自己改，来源见文件头注释） ----------------
COMMISSION = 0.00025      # 佣金 万2.5，买/卖双边各收（券商默认，最低5元对满仓指数可忽略）
STAMP_OLD   = 0.001       # 印花税 0.1%，卖出单边（2023-08-28 之前）
STAMP_NEW   = 0.0005      # 印花税 0.05%，卖出单边（2023-08-28 起，财政部税务总局公告2023年第39号）
STAMP_CUT   = "2023-08-28"
# 过户费 万0.1 双向：0.001% 对 0.025%，小一个量级，忽略。

def stamp_series(dates):
    """每个日期对应的印花税率（卖出单边，分段：2023-08-28 前 0.1%，之后 0.05%）。"""
    cut = pd.Timestamp(STAMP_CUT)
    return np.where(dates >= cut, STAMP_NEW, STAMP_OLD)


def simulate(c, sig, stop_loss):
    """逐日模拟，返回 (每日收盘仓位, 买入日, 卖出日)。

    仓位决策本身没有未来函数：信号 sig 用的是昨日均线（sig[i-1]）。
    收益归因在调用处做（用 pos.shift(1) 还是 pos，决定有没有未来函数）。
    """
    n = len(c)
    position = 0
    entry = 0.0
    pos = np.zeros(n)
    buy = np.zeros(n)
    sell = np.zeros(n)

    for i in range(n):
        price = c[i]
        s = sig[i - 1] if i >= 1 else 0   # 昨天的信号（T+1）
        if position == 0:
            if s == 1:                    # 空仓：昨天金叉了，今天买
                position = 1
                entry = price
                buy[i] = 1
        else:
            if s == 0:                    # 均线死叉 → 卖
                position = 0
                sell[i] = 1
            elif price <= entry * (1 - stop_loss):   # 跌穿红线 → 止损卖
                position = 0
                sell[i] = 1
        pos[i] = position
    return pos, buy, sell


# ---------------- 近十年子区间 ----------------
sub = df[df["date"] >= "2016-01-01"].reset_index(drop=True)
c = sub["close"].to_numpy()
dates = sub["date"].to_numpy()
ma5 = sub["close"].rolling(5).mean().to_numpy()
ma20 = sub["close"].rolling(20).mean().to_numpy()
sig = (ma5 > ma20).astype(int)   # 1=该持有

stamp = stamp_series(dates)                 # 每日印花税率（分段）

# 每日收益（第 i 天 = 从第 i-1 天收盘到第 i 天收盘）
ret = np.zeros(len(c))
ret[1:] = c[1:] / c[:-1] - 1

stops = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
stops_pct = [int(s * 100) for s in stops]

# 滑点旋钮：0 / 0.05% / 0.1% / 0.2%（滑点是假设不是事实，给四档让你自己转）
SLIPPAGE_LEVELS = [0.0, 0.0005, 0.001, 0.002]


def nav_of(held, cost):
    """held=每天吃收益的仓位，cost=每天扣的成本比例 → 总收益（-1 到 +∞）。"""
    strat = held * ret - cost
    return np.prod(1 + strat) - 1


# ---------------- 扫描止损，逐档算各种成本口径 ----------------
rows = []
for stop in stops:
    pos, buy, sell = simulate(c, sig, stop)
    trades = buy + sell                      # 这天发生了一次买卖（买或卖）
    n_trades = int(trades.sum())

    # ① 老幻觉：未来函数 + 单边 0.1%（复现上一课的 1.3 万倍假象）
    cost_old = trades * 0.001
    nav_old = nav_of(pos, cost_old)          # pos 就是未来函数：今天仓位吃今天涨跌

    # ② 修正未来函数：昨天的仓位吃今天的涨跌
    held = np.zeros(len(c))
    held[1:] = pos[:-1]

    # ③ 硬成本 = 佣金(双边) + 印花税(卖出单边)，滑点=0
    cost_hard = trades * COMMISSION + sell * stamp
    nav_hard = nav_of(held, cost_hard)

    # ④ 硬成本 + 各档滑点
    nav_slip = {sl: nav_of(held, cost_hard + trades * sl) for sl in SLIPPAGE_LEVELS}

    # 总摩擦成本（用滑点 0.1% 口径，占本金%，近似=直接相加、忽略复利）
    total_cost_pct = (cost_hard + trades * 0.001).sum() * 100

    rows.append([
        stop * 100,
        n_trades,
        round(total_cost_pct, 1),
        round(nav_old * 100, 1),
        round(nav_hard * 100, 1),
        round(nav_slip[0.001] * 100, 1),
    ])

table = pd.DataFrame(rows, columns=[
    "止损%", "交易次数", "总成本%本金", "老幻觉收益%", "修bug+硬成本%", "再+滑点0.1%%"
])
print("区间：2016-01-01 ~ " + str(df["date"].max().date()) + "（近十年）")
print(table.to_string(index=False))

# ---------------- 画图：一条老幻觉曲线 + 三条真实曲线 ----------------
old_gains = [r[3] for r in rows]
hard_gains = [r[4] for r in rows]
slip_curves = {sl: [] for sl in SLIPPAGE_LEVELS}
for stop in stops:
    pos, buy, sell = simulate(c, sig, stop)
    trades = buy + sell
    held = np.zeros(len(c))
    held[1:] = pos[:-1]
    cost_hard = trades * COMMISSION + sell * stamp
    for sl in SLIPPAGE_LEVELS:
        slip_curves[sl].append(nav_of(held, cost_hard + trades * sl) * 100)

plt.figure(figsize=(11, 6))
plt.plot(stops_pct, old_gains, marker="o", ls="--", color="tab:blue",
         label="老幻觉（未来函数+0.1%成本）")
plt.plot(stops_pct, hard_gains, marker="o", ls="-", color="tab:red",
         label="修正后：佣金+印花税（滑点0）")
plt.plot(stops_pct, slip_curves[0.0005], marker="s", ls="-", color="tab:orange",
         label="+滑点 0.05%")
plt.plot(stops_pct, slip_curves[0.001], marker="s", ls="-", color="tab:green",
         label="+滑点 0.10%")
plt.plot(stops_pct, slip_curves[0.002], marker="s", ls="-", color="tab:purple",
         label="+滑点 0.20%")

plt.axhline(0, color="gray", lw=0.8, alpha=0.6)
plt.xlabel("止损比例（%）")
plt.ylabel("总收益%（对数刻度）")
plt.title("近十年：修掉未来函数 + 加真实成本后，「越小越赚」塌了")
plt.yscale("symlog")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("real_cost_curve.png", dpi=120)

if os.environ.get("Q_HEADLESS"):
    pass
else:
    plt.show()
    input("\n看完按回车键关闭窗口")
