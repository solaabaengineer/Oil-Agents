#!/usr/bin/env python3
"""$OilAgents autonomous trading agent scaffold.

Implements one independent oil-market agent that:
- consumes market snapshots (or generates synthetic snapshots for local testing)
- creates entry/exit decisions with rationale and confidence
- logs decisions to JSONL
- tracks closed and open positions
- computes PnL, win rate, drawdown, and Sharpe ratio
- prints the required end-of-cycle report
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


OIL_ASSETS = [
    "WTI",
    "BRENT",
    "USO",
    "BNO",
    "OIL_TOKEN",
]


@dataclass
class MarketSnapshot:
    asset: str
    price: float
    volume: float
    open_interest: float
    change_1h: float
    volatility_24h: float
    macro_bias: float
    geo_risk: float


@dataclass
class Position:
    id: str
    asset: str
    direction: str
    entry_price: float
    size: float
    entry_time: str
    confidence: float
    expected_timeframe: str
    rationale: str
    entry_cycle: int


@dataclass
class ClosedTrade:
    id: str
    asset: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: str
    exit_time: str
    pnl: float
    return_pct: float
    rationale: str


@dataclass
class DecisionLog:
    timestamp: str
    action: str
    asset: str
    direction: str
    confidence: float
    expected_timeframe: str
    rationale: str


@dataclass
class AgentState:
    capital: float = 100_000.0
    cycle: int = 0
    open_positions: List[Position] = field(default_factory=list)
    closed_trades: List[ClosedTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_state(path: Path) -> AgentState:
    if not path.exists():
        return AgentState()

    data = json.loads(path.read_text(encoding="utf-8"))
    state = AgentState(
        capital=data.get("capital", 100_000.0),
        cycle=data.get("cycle", 0),
        equity_curve=data.get("equity_curve", []),
    )
    state.open_positions = [Position(**p) for p in data.get("open_positions", [])]
    state.closed_trades = [ClosedTrade(**t) for t in data.get("closed_trades", [])]
    return state


def save_state(path: Path, state: AgentState) -> None:
    payload = {
        "capital": state.capital,
        "cycle": state.cycle,
        "open_positions": [asdict(p) for p in state.open_positions],
        "closed_trades": [asdict(t) for t in state.closed_trades],
        "equity_curve": state.equity_curve,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_market_input(path: Optional[Path]) -> List[MarketSnapshot]:
    if path is None:
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    snapshots: List[MarketSnapshot] = []
    for item in data:
        snapshots.append(MarketSnapshot(**item))
    return snapshots


def synthetic_market(agent_seed: int) -> List[MarketSnapshot]:
    rng = random.Random(agent_seed + int(datetime.now(timezone.utc).timestamp() // 300))
    base = {
        "WTI": 79.0,
        "BRENT": 83.0,
        "USO": 76.0,
        "BNO": 30.0,
        "OIL_TOKEN": 1.2,
    }

    snapshots: List[MarketSnapshot] = []
    for asset in OIL_ASSETS:
        price = base[asset] * (1 + rng.uniform(-0.015, 0.015))
        snapshots.append(
            MarketSnapshot(
                asset=asset,
                price=round(price, 5),
                volume=round(rng.uniform(5e5, 4e6), 2),
                open_interest=round(rng.uniform(1e5, 2e6), 2),
                change_1h=round(rng.uniform(-0.02, 0.02), 5),
                volatility_24h=round(rng.uniform(0.01, 0.05), 5),
                macro_bias=round(rng.uniform(-1.0, 1.0), 5),
                geo_risk=round(rng.uniform(-1.0, 1.0), 5),
            )
        )
    return snapshots


def strategy_weights(agent_id: str) -> Dict[str, float]:
    seed = abs(hash(agent_id)) % (10**8)
    rng = random.Random(seed)
    return {
        "momentum": rng.uniform(0.7, 1.5),
        "volume": rng.uniform(0.2, 0.8),
        "open_interest": rng.uniform(0.2, 0.8),
        "macro": rng.uniform(0.4, 1.2),
        "geo": rng.uniform(0.4, 1.2),
        "vol_penalty": rng.uniform(0.5, 1.2),
    }


def confidence_from_score(score: float) -> float:
    # Squashing keeps confidence in [0.50, 0.99] and rewards stronger edges.
    return clamp(0.5 + abs(math.tanh(score)) * 0.49, 0.5, 0.99)


def expected_timeframe(vol_24h: float) -> str:
    if vol_24h < 0.018:
        return "24-72h"
    if vol_24h < 0.032:
        return "8-24h"
    return "1-8h"


def signal_score(snapshot: MarketSnapshot, weights: Dict[str, float]) -> float:
    volume_term = math.tanh((snapshot.volume - 1_500_000.0) / 1_500_000.0)
    oi_term = math.tanh((snapshot.open_interest - 700_000.0) / 700_000.0)
    score = (
        weights["momentum"] * snapshot.change_1h
        + weights["volume"] * volume_term * 0.02
        + weights["open_interest"] * oi_term * 0.02
        + weights["macro"] * snapshot.macro_bias * 0.015
        + weights["geo"] * snapshot.geo_risk * 0.015
        - weights["vol_penalty"] * snapshot.volatility_24h * 0.4
    )
    return score


def position_size(capital: float, price: float, vol_24h: float, confidence: float) -> float:
    risk_fraction = clamp(0.004 + (confidence - 0.5) * 0.02, 0.004, 0.015)
    risk_budget = capital * risk_fraction
    stop_pct = max(0.0075, vol_24h * 0.8)
    units = risk_budget / max(price * stop_pct, 1e-9)
    return round(units, 5)


def mark_to_market(open_positions: List[Position], px_by_asset: Dict[str, float]) -> float:
    unrealized = 0.0
    for pos in open_positions:
        current = px_by_asset[pos.asset]
        delta = current - pos.entry_price
        direction = 1.0 if pos.direction == "LONG" else -1.0
        unrealized += delta * direction * pos.size
    return unrealized


def compute_drawdown(equity_curve: List[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        drawdown = (peak - value) / peak if peak else 0.0
        max_dd = max(max_dd, drawdown)
    return max_dd


def compute_sharpe(closed_trades: List[ClosedTrade]) -> float:
    returns = [t.return_pct / 100.0 for t in closed_trades]
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(len(returns))


def close_position(
    state: AgentState,
    pos: Position,
    exit_price: float,
    reason: str,
    decision_logs: List[DecisionLog],
) -> None:
    direction = 1.0 if pos.direction == "LONG" else -1.0
    pnl = (exit_price - pos.entry_price) * direction * pos.size
    ret_pct = ((exit_price - pos.entry_price) * direction / pos.entry_price) * 100.0

    closed = ClosedTrade(
        id=pos.id,
        asset=pos.asset,
        direction=pos.direction,
        entry_price=pos.entry_price,
        exit_price=exit_price,
        size=pos.size,
        entry_time=pos.entry_time,
        exit_time=utc_now(),
        pnl=round(pnl, 2),
        return_pct=round(ret_pct, 3),
        rationale=reason,
    )
    state.closed_trades.append(closed)

    decision_logs.append(
        DecisionLog(
            timestamp=utc_now(),
            action="EXIT",
            asset=pos.asset,
            direction=pos.direction,
            confidence=pos.confidence,
            expected_timeframe=pos.expected_timeframe,
            rationale=reason,
        )
    )


def append_decisions(log_path: Path, logs: List[DecisionLog]) -> None:
    if not logs:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fp:
        for item in logs:
            fp.write(json.dumps(asdict(item)) + "\n")


def run_cycle(
    state: AgentState,
    snapshots: List[MarketSnapshot],
    agent_id: str,
    entry_threshold: float,
) -> Dict[str, object]:
    state.cycle += 1
    weights = strategy_weights(agent_id)
    now = utc_now()
    decision_logs: List[DecisionLog] = []

    by_asset = {s.asset: s for s in snapshots}
    px = {s.asset: s.price for s in snapshots}

    # Exit logic first: take profit, stop loss, stale position, or strong reversal signal.
    still_open: List[Position] = []
    for pos in state.open_positions:
        snap = by_asset.get(pos.asset)
        if snap is None:
            still_open.append(pos)
            continue

        score = signal_score(snap, weights)
        current = snap.price
        move = (current - pos.entry_price) / pos.entry_price
        directional_move = move if pos.direction == "LONG" else -move
        hold_cycles = state.cycle - pos.entry_cycle

        should_close = False
        reason = ""
        if directional_move >= 0.02:
            should_close = True
            reason = "Take-profit hit (+2% directional move)"
        elif directional_move <= -0.01:
            should_close = True
            reason = "Stop-loss hit (-1% directional move)"
        elif hold_cycles >= 6:
            should_close = True
            reason = "Time-based exit after 6 cycles"
        elif (pos.direction == "LONG" and score < -entry_threshold * 1.2) or (
            pos.direction == "SHORT" and score > entry_threshold * 1.2
        ):
            should_close = True
            reason = "Signal reversal beyond threshold"

        if should_close:
            close_position(state, pos, current, reason, decision_logs)
        else:
            still_open.append(pos)

    state.open_positions = still_open

    # Entry logic: one position per asset.
    open_assets = {p.asset for p in state.open_positions}
    for snap in snapshots:
        if snap.asset in open_assets:
            continue

        score = signal_score(snap, weights)
        if abs(score) < entry_threshold:
            continue

        direction = "LONG" if score > 0 else "SHORT"
        confidence = confidence_from_score(score)
        timeframe = expected_timeframe(snap.volatility_24h)
        size = position_size(state.capital, snap.price, snap.volatility_24h, confidence)

        if size <= 0:
            continue

        pos = Position(
            id=f"{agent_id}-{state.cycle}-{snap.asset}",
            asset=snap.asset,
            direction=direction,
            entry_price=snap.price,
            size=size,
            entry_time=now,
            confidence=round(confidence, 3),
            expected_timeframe=timeframe,
            rationale=(
                f"score={score:.4f}; change_1h={snap.change_1h:.4f}; "
                f"macro={snap.macro_bias:.3f}; geo={snap.geo_risk:.3f}; vol24h={snap.volatility_24h:.3f}"
            ),
            entry_cycle=state.cycle,
        )
        state.open_positions.append(pos)

        decision_logs.append(
            DecisionLog(
                timestamp=utc_now(),
                action="ENTRY",
                asset=snap.asset,
                direction=direction,
                confidence=round(confidence, 3),
                expected_timeframe=timeframe,
                rationale=pos.rationale,
            )
        )

    realized_pnl = sum(t.pnl for t in state.closed_trades)
    unrealized_pnl = mark_to_market(state.open_positions, px)
    live_pnl = realized_pnl + unrealized_pnl
    equity = state.capital + live_pnl
    state.equity_curve.append(round(equity, 2))

    wins = sum(1 for t in state.closed_trades if t.pnl > 0)
    total_closed = len(state.closed_trades)
    win_rate = (wins / total_closed * 100.0) if total_closed else 0.0
    drawdown = compute_drawdown(state.equity_curve)
    sharpe = compute_sharpe(state.closed_trades)

    if snapshots:
        strongest = max(snapshots, key=lambda s: abs(signal_score(s, weights)))
        insight_dir = "upside" if signal_score(strongest, weights) > 0 else "downside"
        insight = (
            f"{strongest.asset} has the strongest {insight_dir} signal into the next 24h "
            f"as macro ({strongest.macro_bias:+.2f}) and geopolitics ({strongest.geo_risk:+.2f}) "
            f"align with intraday momentum ({strongest.change_1h:+.2%})."
        )
    else:
        insight = "No fresh market snapshots were supplied this cycle."

    report = {
        "cycle": state.cycle,
        "timestamp": utc_now(),
        "current_open_positions": [asdict(p) for p in state.open_positions],
        "last_5_closed_trades": [asdict(t) for t in state.closed_trades[-5:]],
        "live_performance_dashboard": {
            "pnl": round(live_pnl, 2),
            "win_rate": round(win_rate, 2),
            "drawdown": round(drawdown * 100.0, 2),
            "sharpe": round(sharpe, 3),
            "closed_trades": total_closed,
            "open_positions": len(state.open_positions),
        },
        "market_insight_next_24h": insight,
        "decision_log_count": len(decision_logs),
    }

    return {"report": report, "decision_logs": decision_logs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one $OilAgents autonomous cycle")
    parser.add_argument("--agent-id", default="agent-1", help="Unique agent identifier")
    parser.add_argument(
        "--state-file",
        default="state/agent-1.json",
        help="Path to persisted state JSON",
    )
    parser.add_argument(
        "--decision-log",
        default="logs/agent-1-decisions.jsonl",
        help="Path to JSONL decision log",
    )
    parser.add_argument(
        "--market-input",
        default="",
        help="Optional JSON file of market snapshots",
    )
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.008,
        help="Absolute signal threshold for entry",
    )
    parser.add_argument(
        "--print-pretty",
        action="store_true",
        help="Pretty-print cycle report JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_file = Path(args.state_file)
    log_file = Path(args.decision_log)
    state = load_state(state_file)

    user_input = Path(args.market_input) if args.market_input else None
    snapshots = read_market_input(user_input) if user_input else synthetic_market(abs(hash(args.agent_id)) % (10**8))

    result = run_cycle(
        state=state,
        snapshots=snapshots,
        agent_id=args.agent_id,
        entry_threshold=args.entry_threshold,
    )

    append_decisions(log_file, result["decision_logs"])
    save_state(state_file, state)

    if args.print_pretty:
        print(json.dumps(result["report"], indent=2))
    else:
        print(json.dumps(result["report"]))


if __name__ == "__main__":
    main()
