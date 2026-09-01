# -*- coding: utf-8 -*-
"""
04_parameter_optimization.py
把「止损比例」当成一个旋钮，从 1% 转到 20%，看收益怎么变。
顺便揭露过拟合：换个时间段，最优参数就变脸。
数据：上证指数（新浪源）
"""

import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = ak.stock_zh_index_daily(symbol="sh000001")
df = df[["date", "close"]].copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

COST = 0.001


def backtest(start, end, stop_loss):
    """跑一次均线策略（带止损），返回 (总收益, 最大回撤)。"""
    sub = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    c = sub["close"].to_numpy()
    ma5 = sub["close"].rolling(5).mean().to_numpy()
    ma20 = sub["close"].rolling(20).mean().to_numpy()
    sig = (ma5 > ma20).astype(int)

    position = 0
    entry = 0.0
    pos = []
    for i in range(len(sub)):
        price = c[i]
        s = sig[i - 1] if i >= 1 else 0
        if position == 0:
            if s == 1:
                position = 1
                entry = price
        else:
            if s == 0:
                position = 0
            elif price <= entry * (1 - stop_loss):
                position = 0
        pos.append(position)

    pos = pd.Series(pos, dtype=float)
    ret = pd.Series(c).pct_change().fillna(0)
    trade = pos.diff().abs().fillna(0)
    strat = pos * ret - trade * COST
    nav = (1 + strat).cumprod()
    return nav.iloc[-1] - 1, (nav / nav.cummax() - 1).min()


# 待试的止损比例：1% 到 20%
stops = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]
stops_pct = [int(s * 100) for s in stops]

# 两个时间段：全程 vs 近十年
windows = [
    ("全程 1990-2026", "1990-12-19", "2026-08-14"),
    ("近十年 2016-2026", "2016-01-01", "2026-08-14"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes = axes.flatten()

for i, (label, s, e) in enumerate(windows):
    gains = []
    dds = []
    for pct in stops:
        g, dd = backtest(s, e, pct)
        gains.append(g * 100)
        dds.append(dd * 100)

    ax = axes[i]
    ax.plot(stops_pct, gains, marker="o", color="tab:red", label="总收益%")
    ax.set_title(label)
    ax.set_xlabel("止损比例（%）")
    ax.set_ylabel("总收益%")
    ax.grid(True, alpha=0.3)

    # 用对数刻度，否则「1.3 万倍」和「-10%」没法画在一张图上
    ax.set_yscale("symlog")
    ax.legend()

    best = stops_pct[gains.index(max(gains))]
    print(f"{label}：收益最高的止损 = {best}%，收益 = {max(gains):.1f}%")

plt.suptitle("止损参数扫描：收益越高不代表越真实（注意 y 轴是对数刻度）", fontsize=13)
plt.tight_layout()
plt.savefig("parameter_optimization.png", dpi=120)
plt.show()

input("\n看完按回车键关闭窗口")
