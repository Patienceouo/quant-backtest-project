# Quantitative Backtest Project

A self-initiated Python project applying technical-analysis strategies and backtesting to China's SSE Composite Index — built to learn the fundamentals of systematic trading and to validate (and challenge) intuition with data.

## What I Did

| Script | What it does |
|---|---|
| `01_ma_strategy_backtest.py` | Golden-cross / dead-cross moving-average strategy on synthetic data |
| `02_window_backtest.py` | Backtested the strategy across 6 market regimes (2006–2024) on real SSE Composite Index data |
| `03_stop_loss.py` | Added stop-loss rules; measured impact on return and max drawdown |
| `04_parameter_optimization.py` | Swept stop-loss parameters to study return sensitivity and overfitting |
| `05_real_cost.py` | Fixed look-ahead bias + added real transaction costs; debunked the "tighter stop = more profit" illusion |
| `06_intraday_stop.py` | Modeled real stop-loss execution (intraday trigger, gap-downs, limit-lock days) — final proof that stop-loss can't save the MA strategy |

## Tools

Python · pandas · numpy · matplotlib · akshare

## Key Findings

- **Stop-loss improved both return and drawdown** — a 5% stop-loss lifted total return from 160x to 618x while cutting max drawdown from -57% to -38% (counter-intuitive, confirmed by backtest).
- **Parameters overfit to specific historical periods** — an "optimal" parameter tuned on one market regime failed on others. The takeaway was to split data into training / validation / test sets.
- **Real costs flip the result** — fixing look-ahead bias and adding commission + stamp duty + slippage turned "1% stop wins" into "every stop level loses".
- **Execution assumptions matter** — intraday triggering, gap-downs, and limit-lock days were modeled one by one; the strategy still loses at every stop level. The reusable takeaway is a falsification method: future function → costs → execution.

## Data

SSE Composite Index (1990–2026), retrieved via akshare.

## Next Step

Preventing overfitting with a three-way data split (tuning / validation / test), and porting the falsification method to individual stocks (limit-up/down, suspension, delisting, survivorship bias).
