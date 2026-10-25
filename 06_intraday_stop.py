# -*- coding: utf-8 -*-
"""
06_intraday_stop.py
第 6 课：盘中止损——分三刀，把「假止损」一层层改成「真止损」。

第 5 课判断止损用的是「收盘价 close」：
    if 收盘价 <= 买入价 * (1 - 止损)   → 卖
这等于假设「价格只在收盘那一刻才看一眼」。真止损单是「价格一到就触发」。

第 1 刀：判断依据 close → low（盘中触发）。
第 2 刀：触发后的成交价 close → 止损价（价格一到就按止损价卖，不再假装拖到收盘）。
第 3 刀：跳空 + 封死——
    (1) 跳空日 open 已低于止损线 → 只能按开盘价成交；
    (2) 单日跌幅 >= 7%（近似熔断停市/全线跌停）→ 当天卖不掉，顺延次日开盘卖。

本脚本输出四列对比：
    第 1 列 = 收盘判断 + 收盘成交（第5课，最假）
    第 2 列 = low 触发 + 收盘成交（第1刀，只改触发器）
    第 3 列 = low 触发 + 止损价成交（第2刀，触发器+成交价都改）
    第 4 列 = 第3刀：跳空按开盘价 + 封死顺延（触发器+成交价+跳空+封死全改）

数据：上证指数（新浪源 stock_zh_index_daily，有 open/high/low/close 四个价）。
"""
import os
import numpy as np
import pandas as pd
import akshare as ak
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 拉数据（新浪源，这次四个价都取） ----------------
df = ak.stock_zh_index_daily(symbol="sh000001")
df = df[["date", "open", "high", "low", "close"]].copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------------- 真实成本（同第 5 课） ----------------
COMMISSION = 0.00025      # 佣金 万2.5，买/卖双边
STAMP_OLD   = 0.001       # 印花税 0.1%（2023-08-28 之前，卖出单边）
STAMP_NEW   = 0.0005      # 印花税 0.05%（2023-08-28 起，卖出单边）
STAMP_CUT   = "2023-08-28"


def stamp_series(dates):
    cut = pd.Timestamp(STAMP_CUT)
    return np.where(dates >= cut, STAMP_NEW, STAMP_OLD)


# ---------------- 近十年子区间 ----------------
sub = df[df["date"] >= "2016-01-01"].reset_index(drop=True)
close = sub["close"].to_numpy()
low = sub["low"].to_numpy()
op = sub["open"].to_numpy()
dates = sub["date"].to_numpy()
ma5 = sub["close"].rolling(5).mean().to_numpy()
ma20 = sub["close"].rolling(20).mean().to_numpy()
sig = (ma5 > ma20).astype(int)        # 信号仍用收盘价算均线（不变）

stamp = stamp_series(dates)

# 每日收益（第 i 天 = 从第 i-1 天收盘到第 i 天收盘）
ret = np.zeros(len(close))
ret[1:] = close[1:] / close[:-1] - 1

stops = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
stops_pct = [int(s * 100) for s in stops]


LIMIT_FREEZE = 0.07   # 「卖不掉」近似阈值：单日跌幅>=7%（熔断停市线），第3刀用


def simulate(close, low, open_, sig, stop_loss,
             use_low=False, exit_at_stop=False, use_gap=False):
    """逐日模拟。

    use_low=True      ：止损用当日最低价(low)判断（盘中触发）；
    exit_at_stop=True ：止损成交价 = 止损线(entry*(1-stop))，否则 = 收盘价(close)。
    use_gap=True      ：第3刀——跳空日(open<止损线)只能按开盘价成交；
                        单日跌幅>=LIMIT_FREEZE 当天卖不掉，顺延次日开盘卖。
    """
    n = len(close)
    position = 0
    entry = 0.0
    pending = False                          # 昨天封死没卖掉 → 今天开盘必须卖
    pos = np.zeros(n)
    buy = np.zeros(n)
    sell = np.zeros(n)
    fill = np.full(n, np.nan)                # 卖出日的实际成交价（其余为 nan）

    for i in range(n):
        price = close[i]
        s = sig[i - 1] if i >= 1 else 0      # 昨天的信号（T+1）
        if position == 0:
            if s == 1:                       # 空仓：昨天金叉了，今天买
                position = 1
                entry = price
                buy[i] = 1
        else:
            if pending:
                # 第3刀：昨天封死没卖掉，今天开盘市价卖
                if open_[i] / close[i - 1] - 1 <= -LIMIT_FREEZE:
                    pass                     # 开盘又封死，继续顺延（仓位还挂着）
                else:
                    position = 0
                    sell[i] = 1
                    fill[i] = open_[i]
                    pending = False
            elif s == 0:                     # 均线死叉 → 卖（按收盘价，第3刀不动它）
                position = 0
                sell[i] = 1
                fill[i] = price
            else:
                # 判断止损的依据，这里分叉：
                trigger = low[i] if use_low else price
                if trigger <= entry * (1 - stop_loss):
                    stop_price = entry * (1 - stop_loss)
                    if use_gap:
                        # 第3刀改①：跳空日 open 已跌破止损线 → 只能按 open 卖
                        fill_price = min(stop_price, open_[i])
                    elif exit_at_stop:
                        fill_price = stop_price      # 第2刀：价格一到就成交
                    else:
                        fill_price = price           # 第1刀：还按收盘价
                    if use_gap and close[i] / close[i - 1] - 1 <= -LIMIT_FREEZE:
                        # 第3刀改②：当天封死，卖不掉 → 挂起，顺延
                        pending = True
                    else:
                        position = 0
                        sell[i] = 1
                        fill[i] = fill_price
        pos[i] = position
    return pos, buy, sell, fill


def nav_of(held, cost, ret_eff=None):
    """held=每天吃收益的仓位，cost=每天扣的成本比例 → 总收益。

    ret_eff 省略时用全局 ret（收盘到收盘）；卖出日可换成「成交价收益」。
    """
    if ret_eff is None:
        ret_eff = ret
    strat = held * ret_eff - cost
    return np.prod(1 + strat) - 1


def calc(pos, buy, sell, fill):
    """给定一次模拟结果，算交易次数 + 两种成本口径的收益。

    fill=卖出日的实际成交价（nan=不是卖出日）。卖出日当天的收益
    用「成交价 / 前一日收盘」算，而不是收盘到收盘——这是第2刀的改动。
    """
    trades = buy + sell
    n_trades = int(trades.sum())
    held = np.zeros(len(close))
    held[1:] = pos[:-1]                       # 昨天的仓位，吃今天的涨跌（修未来函数）
    ret_eff = ret.copy()
    sold_idx = np.where(~np.isnan(fill))[0]
    if len(sold_idx):
        ret_eff[sold_idx] = fill[sold_idx] / close[sold_idx - 1] - 1
    cost_hard = trades * COMMISSION + sell * stamp
    nav_hard = nav_of(held, cost_hard, ret_eff)
    nav_slip = nav_of(held, cost_hard + trades * 0.001, ret_eff)   # 再 +0.1% 滑点
    return n_trades, nav_hard * 100, nav_slip * 100


# ---------------- 扫描止损，逐档对比「收盘判断 vs 盘中判断」 ----------------
rows = []
base_curve = []   # 收盘判断 + 收盘成交（第5课口径）
low_curve = []    # low 触发 + 收盘成交（第1刀）
fill_curve = []   # low 触发 + 止损价成交（第2刀）
gap_curve = []    # 第3刀：跳空按开盘价 + 封死顺延

for stop in stops:
    pos0, buy0, sell0, fill0 = simulate(close, low, op, sig, stop, use_low=False, exit_at_stop=False)
    pos1, buy1, sell1, fill1 = simulate(close, low, op, sig, stop, use_low=True,  exit_at_stop=False)
    pos2, buy2, sell2, fill2 = simulate(close, low, op, sig, stop, use_low=True,  exit_at_stop=True)
    pos3, buy3, sell3, fill3 = simulate(close, low, op, sig, stop, use_low=True,  exit_at_stop=True, use_gap=True)

    n0, hard0, slip0 = calc(pos0, buy0, sell0, fill0)
    n1, hard1, slip1 = calc(pos1, buy1, sell1, fill1)
    n2, hard2, slip2 = calc(pos2, buy2, sell2, fill2)
    n3, hard3, slip3 = calc(pos3, buy3, sell3, fill3)

    rows.append([stop * 100,
                 n0, round(slip0, 1),
                 n1, round(slip1, 1),
                 n2, round(slip2, 1),
                 n3, round(slip3, 1)])

    base_curve.append(slip0)
    low_curve.append(slip1)
    fill_curve.append(slip2)
    gap_curve.append(slip3)

table = pd.DataFrame(rows, columns=[
    "止损%", "收盘:次数", "收盘:+滑点%",
             "low触发:次数", "low触发:+滑点%",
             "low+止损价成交:次数", "low+止损价成交:+滑点%",
             "第3刀跳空+封死:次数", "第3刀跳空+封死:+滑点%"
])
print("区间：2016-01-01 ~ " + str(df["date"].max().date()) + "（近十年）")
print("口径：都修过未来函数、加过佣金+印花税+滑点。")
print("第1列=收盘判断；第2列=low触发；第3列=low触发+止损价成交；第4列=第3刀跳空按开盘价+封死顺延。")
print(table.to_string(index=False))

# ---------------- 画图：四条曲线对比 ----------------
plt.figure(figsize=(10, 6))
plt.plot(stops_pct, base_curve, marker="o", ls="--", color="tab:red",
         label="收盘判断+收盘成交（第5课）")
plt.plot(stops_pct, low_curve, marker="o", ls=":", color="tab:green",
         label="low触发+收盘成交（第1刀）")
plt.plot(stops_pct, fill_curve, marker="o", ls="-", color="tab:blue",
         label="low触发+止损价成交（第2刀）")
plt.plot(stops_pct, gap_curve, marker="o", ls="-", color="tab:purple",
         label="第3刀：跳空按开盘价+封死顺延")
plt.axhline(0, color="gray", lw=0.8, alpha=0.6)
plt.xlabel("止损比例（%）")
plt.ylabel("总收益%（对数刻度）")
plt.title("盘中止损第3刀：跳空日按开盘价成交，封死日顺延次日")
plt.yscale("symlog")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("knife3_gap_stop.png", dpi=120)

if os.environ.get("Q_HEADLESS"):
    pass
else:
    plt.show()
    input("\n看完按回车键关闭窗口")
