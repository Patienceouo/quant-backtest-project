# -*- coding: utf-8 -*-
"""
02_window_backtest.py
同一套 5/20 日均线策略，换不同年份区间回测，看结果怎么变。
数据：上证指数（新浪源，免费）
"""

import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt

# 让图能正常显示中文（Windows 自带微软雅黑/黑体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False   # 让负号 "-" 正常显示

# 1) 拉真实数据：上证指数日线（1990 年至今）
df = ak.stock_zh_index_daily(symbol="sh000001")
df = df[["date", "close"]].copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 交易成本：单边 0.1%（和第 1 课一致，买/卖各扣一次）
COST = 0.001


def backtest(start, end):
    """在 [start, end] 区间内跑 5/20 均线金叉死叉策略。"""
    sub = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    if len(sub) < 30:
        return None

    c = sub["close"]
    ma5 = c.rolling(5).mean()
    ma20 = c.rolling(20).mean()

    signal = (ma5 > ma20).astype(int)            # 1=持有, 0=空仓
    position = signal.shift(1).fillna(0)         # T+1：今天信号明天才生效

    daily_ret = c.pct_change().fillna(0)         # 每天涨跌 %
    trade = position.diff().abs().fillna(0)      # 1=这天发生了一次买卖

    strat_ret = position * daily_ret - trade * COST
    nav = (1 + strat_ret).cumprod()              # 策略净值（1 元滚到今天）
    hold_nav = (1 + daily_ret).cumprod()         # 买入持有净值（对照）

    total_ret = nav.iloc[-1] - 1
    max_dd = (nav / nav.cummax() - 1).min()      # 最大回撤

    return {
        "收益": round(total_ret * 100, 2),
        "回撤": round(max_dd * 100, 2),
        "交易": int(trade.sum()),
        "持有": round((hold_nav.iloc[-1] - 1) * 100, 2),
        "nav": nav,                              # 用于画图
        "hold_nav": hold_nav,                    # 用于画图
        "date": sub["date"],                     # 用于画图
    }


# 2) 六个「不同天气」的区间：牛市 / 熊市 / 震荡 / 全程
windows = [
    ("2006-2007 大牛市",   "2006-01-01", "2007-12-31"),
    ("2008 熊市",          "2008-01-01", "2008-12-31"),
    ("2015-2016 股灾熔断", "2015-01-01", "2016-12-31"),
    ("2019-2021 结构牛",   "2019-01-01", "2021-12-31"),
    ("2022-2024 震荡",     "2022-01-01", "2024-12-31"),
    ("全程",               str(df["date"].min().date()), str(df["date"].max().date())),
]

# 3) 每个区间跑一遍，结果存起来
results = []
for name, s, e in windows:
    r = backtest(s, e)
    if r is not None:
        results.append((name, r))

# 4) 数字表格
rows = [[name, r["收益"], r["回撤"], r["交易"], r["持有"]] for name, r in results]
table = pd.DataFrame(rows, columns=["区间", "策略收益%", "最大回撤%", "交易次数", "买入持有%"])
print(table.to_string(index=False))

# 5) 画图：6 个子图，每个区间一条净值曲线（红=策略，蓝=买入持有）
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, (name, r) in enumerate(results):
    ax = axes[i]
    ax.plot(r["date"], r["nav"], lw=1.3, color="tab:red", label="策略")
    ax.plot(r["date"], r["hold_nav"], lw=1.0, color="tab:blue", alpha=0.7, label="买入持有")
    ax.set_title(f"{name}：策略 {r['收益']}%  vs  买入持有 {r['持有']}%", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle("同一套 5/20 均线策略，换不同年份区间，结果天差地别", fontsize=14)
plt.tight_layout()
plt.savefig('window_compare.png', dpi=120)   # 顺便存一张图片，方便你直接看
plt.show()

input("\n看完按回车键关闭窗口")
