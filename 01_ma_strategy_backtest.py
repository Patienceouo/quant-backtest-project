# -*- coding: utf-8 -*-
"""
均线策略 + 回测 —— 零基础第一课
用本地模拟数据跑通「量化到底在干嘛」的完整流程。

量化四步：数据 -> 信号 -> 回测 -> 评估
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 让图能正常显示中文（Windows 自带微软雅黑/黑体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False   # 让负号 "-" 正常显示

# ========== 第 1 步：数据（本地模拟，不联网） ==========
# 固定随机种子 -> 每次运行结果完全一样（可复现）
np.random.seed(42)

n_days = 250                              # 250 个交易日 ≈ 一年
daily_ret = np.random.normal(0.0005, 0.02, n_days)  # 每日涨跌：均值 0.05%、波动 2%
price = 100 * np.exp(np.cumsum(daily_ret))          # 从 100 元开始，累计成股价

df = pd.DataFrame({'close': price})        # 存成表格，格式和真实行情一样

# ========== 第 2 步：算均线 + 生成买卖信号 ==========
SHORT, LONG = 5, 20                        # 短均线 5 天，长均线 20 天

df['ma_short'] = df['close'].rolling(SHORT).mean()   # 5 日均线
df['ma_long']  = df['close'].rolling(LONG).mean()    # 20 日均线

# 信号：短期均线在长期均线上方 -> 持有(1)；下方 -> 空仓(0)
df['signal'] = (df['ma_short'] > df['ma_long']).astype(int)

# ========== 第 3 步：回测（逐日模拟买卖） ==========
# 交易成本：每次买入/卖出各收一次（单边）。0.1% = 佣金 + 滑点，合并的教学值
COST = 0.001

df['ret'] = df['close'].pct_change().fillna(0)        # 每天涨跌幅

# 关键点：今天的信号，明天才能下单执行（shift 一天）
# 这样避免「未来函数」——不能用今天收盘才看到的信息在今天买入
df['pos'] = df['signal'].shift(1).fillna(0)           # 实际持仓：0=空仓, 1=满仓

# 交易发生的日子：持仓从 0->1(买) 或 1->0(卖)，这一天要交手续费
df['trade'] = df['pos'].diff().abs().fillna(0)        # 1=当天有交易, 0=没有

# 策略收益（无成本）= 持仓 × 当天涨跌
df['strategy_ret'] = df['pos'] * df['ret']

# 策略收益（有成本）= 无成本收益 - 当天手续费
df['strategy_ret_cost'] = df['strategy_ret'] - df['trade'] * COST

# 资金曲线：1 元本金，复利累计
df['equity_no_cost'] = (1 + df['strategy_ret']).cumprod()       # 策略·无成本
df['equity_cost']    = (1 + df['strategy_ret_cost']).cumprod()  # 策略·有成本
df['equity_buyhold'] = (1 + df['ret']).cumprod()                # 对照组：一直满仓

# ========== 第 4 步：评估 ==========
total_no_cost = df['equity_no_cost'].iloc[-1] - 1
total_cost    = df['equity_cost'].iloc[-1] - 1
total_buyhold = df['equity_buyhold'].iloc[-1] - 1
n_trades = int((df['signal'].diff().abs() == 1).sum())   # 信号变化一次 = 一次买卖

print("=" * 40)
print(f"Strategy WITHOUT cost : {total_no_cost:.2%}")
print(f"Strategy WITH cost    : {total_cost:.2%}")
print(f"Buy & Hold return     : {total_buyhold:.2%}")
print(f"Number of trades      : {n_trades}")
print(f"Cost ate away         : {total_no_cost - total_cost:.2%}")
print("=" * 40)

# ========== 画图 ==========
# 买卖点：signal 从 0->1 是买入(金叉)，从 1->0 是卖出(死叉)
buy_idx  = df.index[df['signal'].diff() == 1]
sell_idx = df.index[df['signal'].diff() == -1]

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 背景色：标记"实际持仓"区间（浅红=满仓，浅灰=空仓）
# 实际持仓用前一天信号(shift 1)，和资金曲线的算法完全一致
pos = df['signal'].shift(1).fillna(0).astype(int)   # 0=空仓, 1=满仓
grp = (pos != pos.shift()).cumsum()                 # 给每个连续区间编个号
for _, g in df.groupby(grp):
    s, e = int(g.index[0]), int(g.index[-1])
    color = '#ffd9d9' if pos.loc[g.index[0]] == 1 else '#ececec'
    for ax in axes:
        ax.axvspan(s, e, color=color, alpha=0.5, zorder=0)

# 上图：股价 + 两条均线 + 买卖点
axes[0].plot(df['close'], label='股价', lw=1.2)
axes[0].plot(df['ma_short'], label='5日均线(短线)', lw=1)
axes[0].plot(df['ma_long'], label='20日均线(长线)', lw=1)
axes[0].scatter(buy_idx, df['close'].loc[buy_idx], marker='^', color='red', s=80, label='买入')
axes[0].scatter(sell_idx, df['close'].loc[sell_idx], marker='v', color='green', s=80, label='卖出')
axes[0].legend()
axes[0].set_title('上图：股价 + 均线 + 买卖信号（浅红=满仓，浅灰=空仓）')
axes[0].set_ylabel('价格(元)')

# 下图：资金曲线对比（无成本 vs 有成本 vs 买入持有）
axes[1].plot(df['equity_no_cost'], label='策略·无成本', lw=1.5, linestyle='--')
axes[1].plot(df['equity_cost'], label='策略·有成本', lw=1.5)
axes[1].plot(df['equity_buyhold'], label='买入持有', lw=1.5, alpha=0.7)
axes[1].legend()
axes[1].set_title('下图：资金曲线(起始本金 = 1 元)')
axes[1].set_ylabel('资金(元)')
axes[1].set_xlabel('交易日')

plt.tight_layout()
plt.savefig('backtest_result.png', dpi=120)
plt.show()
