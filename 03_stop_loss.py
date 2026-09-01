# -*- coding: utf-8 -*-
"""
03_stop_loss.py
给 5/20 均线策略加一道「止损」，看最大回撤能不能压下来。
数据：上证指数（新浪源）
"""

import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt

# 让图能正常显示中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 拉数据（上证指数日线）
df = ak.stock_zh_index_daily(symbol="sh000001")
df = df[["date", "close"]].copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

COST = 0.001   # 单边交易成本 0.1%


def backtest(start, end, stop_loss=None):
    """
    在 [start, end] 区间跑均线策略。
    stop_loss：止损比例。None=不设止损；0.08=跌 8% 就割肉离场。
    """
    sub = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    c = sub["close"]
    ma5 = c.rolling(5).mean()
    ma20 = c.rolling(20).mean()

    # 均线信号：1=金叉状态(该持有)，0=死叉状态(该空仓)
    signal = (ma5 > ma20).astype(int)

    # 因为止损要「记住买入价」，只能一天一天模拟（for 循环）
    position = 0        # 当前持仓：0=空仓，1=满仓
    entry = 0.0         # 买入价
    pos_list = []       # 记录每天的实际持仓

    for i in range(len(sub)):
        price = c.iloc[i]
        sig = signal.iloc[i - 1] if i >= 1 else 0   # 昨天的信号（T+1，防未来函数）

        if position == 0:
            if sig == 1:                  # 空仓：昨天金叉了，今天就买入
                position = 1
                entry = price
        else:
            if sig == 0:                  # 均线死叉 → 正常卖出
                position = 0
            elif stop_loss is not None and price <= entry * (1 - stop_loss):
                position = 0              # 跌穿红线 → 割肉止损

        pos_list.append(position)

    pos = pd.Series(pos_list, index=sub.index)
    daily_ret = c.pct_change().fillna(0)
    trade = pos.diff().abs().fillna(0)    # 1=这天发生一次买卖
    strat_ret = pos * daily_ret - trade * COST

    nav = (1 + strat_ret).cumprod()       # 净值曲线
    total = nav.iloc[-1] - 1
    dd = (nav / nav.cummax() - 1).min()   # 最大回撤

    return {
        "total": round(total * 100, 2),
        "dd": round(dd * 100, 2),
        "trades": int(trade.sum()),
        "nav": nav,                       # 用于画图
        "date": sub["date"],              # 用于画图
    }


# 三种情况：无止损 / 8% 止损 / 5% 止损（全程 36 年）
start = str(df["date"].min().date())
end = str(df["date"].max().date())

configs = [
    (None,  "无止损",  "tab:green"),
    (0.08, "止损 8%", "tab:orange"),
    (0.05, "止损 5%", "tab:red"),
]

results = []
for stop, label, color in configs:
    r = backtest(start, end, stop)
    results.append((label, color, r))

# 数字表格
rows = [[label, r["total"], r["dd"], r["trades"]] for label, _, r in results]
table = pd.DataFrame(rows, columns=["设置", "总收益%", "最大回撤%", "交易次数"])
print(f"区间：{start} ~ {end}（全程）")
print(table.to_string(index=False))

# 画图：三条净值曲线叠在一张图上
plt.figure(figsize=(13, 6))
for label, color, r in results:
    plt.plot(r["date"], r["nav"], lw=1.3, color=color, label=label)

# 36 年复利，倍数差太大，用对数刻度才看得清早期（否则早期都挤成一条线）
plt.yscale("log")
plt.title("加止损 vs 不加止损：上证指数全程净值曲线")
plt.xlabel("日期")
plt.ylabel("净值（元，对数刻度）")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("stop_loss_compare.png", dpi=120)
plt.show()

input("\n看完按回车键关闭窗口")
