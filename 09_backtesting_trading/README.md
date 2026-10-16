# Backtesting and Trading Simulation

Evaluates prediction-driven trading strategies against historical data.

## Files

| File | Purpose |
| --- | --- |
| `backtesting_framework.py` | Main framework — loads predictions, generates signals, simulates trades, computes Sharpe ratio, maximum drawdown, win rate, and comparison with Buy & Hold |
| `live_trading_engine.py` | Simulated live trading engine — polls for new predictions, generates orders, tracks positions and P&L |
| `quick_backtest.py` | Smoke test that runs the framework end-to-end with a small dataset |
| `quick_test.py` | Smoke test that exercises the live trading engine |

## Legacy

`legacy/` contains:

- `live_trading_engine.py` — v1 (superseded by the fixed version)
- `balanced_production.py` — a specific production configuration variant

## Metrics

The framework computes:

- Sharpe ratio
- Maximum drawdown
- Win rate
- Cumulative performance
- Strategy returns vs. Buy & Hold

## Status

**Not applied to live trading.** All backtests are historical
simulations. Results do not account for transaction costs, slippage, or
market impact.
