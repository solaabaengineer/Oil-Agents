# Oil Agents

![image - 2026-03-08T235816 490](https://github.com/user-attachments/assets/dbe0d89e-8fa7-4de8-92a9-844a902e74ea)

https://x.com/OilAgents


Autonomous oil-market trading agent scaffold for the `$OilAgents` competition.

This repository contains a runnable Python agent that executes real-time style cycles,
generates entry/exit signals, logs decisions, manages open positions, and publishes a
standardized performance dashboard.

## What This Agent Does

- Operates only on oil-linked instruments:
	- `WTI`, `BRENT`, `USO`, `BNO`, `OIL_TOKEN`
- Evaluates each cycle using:
	- price action (`change_1h`)
	- volume
	- open interest
	- macro bias
	- geopolitical risk
	- volatility
- Creates entries/exits with rationale and confidence
- Logs all decisions with timestamp, asset, direction, confidence, timeframe
- Tracks performance metrics:
	- PnL
	- win rate
	- max drawdown
	- Sharpe ratio

## Files

- `oil_agent.py`: main agent logic and CLI
- `example_market_input.json`: sample oil market snapshots
- `state/`: persisted agent state (created at runtime)
- `logs/`: JSONL decision logs (created at runtime)

## Requirements

- Python 3.10+

## Quick Start

Run one cycle with synthetic market data:

```bash
python3 oil_agent.py --agent-id agent-1 --print-pretty
```

Run one cycle with explicit market input:

```bash
python3 oil_agent.py \
	--agent-id agent-1 \
	--state-file state/agent-1.json \
	--decision-log logs/agent-1-decisions.jsonl \
	--market-input example_market_input.json \
	--print-pretty
```

Run multiple agents independently (divergent strategies via `agent-id`):

```bash
python3 oil_agent.py --agent-id agent-1 --state-file state/a1.json --decision-log logs/a1.jsonl
python3 oil_agent.py --agent-id agent-2 --state-file state/a2.json --decision-log logs/a2.jsonl
python3 oil_agent.py --agent-id agent-3 --state-file state/a3.json --decision-log logs/a3.jsonl
python3 oil_agent.py --agent-id agent-4 --state-file state/a4.json --decision-log logs/a4.jsonl
python3 oil_agent.py --agent-id agent-5 --state-file state/a5.json --decision-log logs/a5.jsonl
```

## End-of-Cycle Output Format

Each cycle prints JSON with:

- `current_open_positions`
- `last_5_closed_trades`
- `live_performance_dashboard` (`pnl`, `win_rate`, `drawdown`, `sharpe`)
- `market_insight_next_24h`

This matches the required `$OilAgents` output contract.

## Notes

- This is a protocol-ready scaffold with deterministic divergence by `agent-id`.
- Replace synthetic/input data plumbing with your real-time feed handlers to run live.
