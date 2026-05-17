"""Portfolio simulator.

Given a portfolio definition (tickers, weights, signal source, rules), this
module produces:

* a rich list of decisions (one per ticker per day, regardless of execution),
* an even richer list of executed transactions (the subset of decisions where
  the portfolio actually changed),
* the daily equity curve,
* the final positions snapshot,
* metrics (portfolio and trade level).

The crucial design point is that *signal*, *decision*, and *execution* are
three distinct concepts that we keep separate by storing IDs that link them:

    signal_id        — emitted by ``signals`` module; one per (ticker, date, model).
    decision_id      — emitted here; one per (portfolio, ticker, date).
    transaction_id   — emitted here; one per actual trade.

So one signal can produce zero or one decisions, and one decision can produce
zero or one transactions (HOLD decisions produce no transaction).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import metrics as metrics_mod

logger = logging.getLogger("dashboard.portfolio")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _short_uuid() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

@dataclass
class Position:
    ticker: str
    shares: float = 0.0
    cost_basis: float = 0.0   # average price paid for current shares
    entry_date: Optional[str] = None
    entry_price: Optional[float] = None
    last_signal_id: Optional[str] = None


@dataclass
class BacktestResult:
    portfolio_id: str
    portfolio_name: str
    description: str
    initial_capital: float
    final_value: float
    cash: float
    transactions: List[Dict] = field(default_factory=list)
    decisions: List[Dict] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    positions: List[Dict] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    benchmark_metrics: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


def run_portfolio_backtest(
    portfolio_cfg: Dict,
    prices_by_ticker: Dict[str, pd.DataFrame],
    signals_by_ticker_model: Dict[str, Dict[str, pd.DataFrame]],
    benchmark_prices: Optional[pd.DataFrame] = None,
    data_version: str = "unversioned",
    strategy_version: str = "1.0",
    label_mode: str = "binary",
    label_version: str = "binary-2026-05",
) -> BacktestResult:
    """Simulate the portfolio day-by-day and return a full audit trail.

    ``signals_by_ticker_model[ticker][model_name]`` must be a DataFrame indexed
    by date with columns produced by :mod:`signals` (signal, score,
    probability_up, confidence, threshold_used, triggered_rules).

    ``prices_by_ticker[ticker]`` must be a DataFrame with OHLCV + indicator
    columns (Open/High/Low/Close/Volume + RSI/MACD/etc).
    """
    pid = portfolio_cfg["id"]
    tickers = [t for t in portfolio_cfg["tickers"] if t in prices_by_ticker]
    if not tickers:
        return BacktestResult(
            portfolio_id=pid,
            portfolio_name=portfolio_cfg["name"],
            description=portfolio_cfg.get("description", ""),
            initial_capital=portfolio_cfg["initial_capital"],
            final_value=portfolio_cfg["initial_capital"],
            cash=portfolio_cfg["initial_capital"],
            metadata={"warning": "no tickers with data available"},
        )

    signal_source = portfolio_cfg.get("signal_source", "technical_rule_based")
    rules = portfolio_cfg.get("rules", {}) or {}

    # Determine the calendar — union of all ticker indices, then sort.
    all_dates = sorted(set().union(*(prices_by_ticker[t].index for t in tickers)))

    # Equal-weight target per slot (max_positions or len(tickers))
    max_positions = int(rules.get("max_positions", len(tickers)))
    n_slots = max(1, min(max_positions, len(tickers)))
    target_weight = 1.0 / n_slots

    cash = float(portfolio_cfg["initial_capital"])
    initial_capital = cash
    positions: Dict[str, Position] = {t: Position(ticker=t) for t in tickers}

    transactions: List[Dict] = []
    decisions: List[Dict] = []
    equity_curve: List[Dict] = []
    trade_pnls: List[float] = []

    generated_at = _utc_now_iso()

    for d in all_dates:
        # 1) Mark-to-market: portfolio value at *open* of day d using prev close
        portfolio_value_before = cash + sum(
            positions[t].shares * _price_on(prices_by_ticker[t], d, "Close")
            for t in tickers
            if positions[t].shares > 0
        )

        for t in tickers:
            df = prices_by_ticker[t]
            if d not in df.index:
                continue

            # Decide the signal for the day based on the configured source
            chosen_signal = _resolve_signal_for_day(
                signal_source=signal_source,
                ticker=t,
                day=d,
                signals_by_model=signals_by_ticker_model.get(t, {}),
                rules=rules,
            )
            if chosen_signal is None:
                continue
            sig_row = chosen_signal["row"]
            sig_model = chosen_signal["model_name"]
            sig_id = chosen_signal["signal_id"]

            row = df.loc[d]
            execution_price = float(row["Open"]) if pd.notna(row.get("Open")) else float(row["Close"])
            prev_close = _previous_close(df, d)

            pos = positions[t]
            shares_before = pos.shares
            cash_before = cash
            pos_val_before = shares_before * execution_price
            weight_actual_before = pos_val_before / portfolio_value_before if portfolio_value_before > 0 else 0

            action = "HOLD"
            quantity = 0.0
            reason = sig_row.get("triggered_rules", "")
            decision_id = f"D-{pid}-{_short_uuid()}"

            sig = str(sig_row["signal"])
            # Apply portfolio rules to decide whether the signal turns into a trade
            if sig == "BUY" and shares_before == 0 and cash > 0:
                # Compute target $ allocation; cap by available cash
                target_dollars = portfolio_value_before * target_weight if portfolio_value_before > 0 else cash * target_weight
                target_dollars = min(target_dollars, cash)
                if target_dollars >= execution_price > 0:
                    quantity = math_floor(target_dollars / execution_price)
                    if quantity > 0:
                        action = "BUY"
            elif sig == "SELL" and shares_before > 0:
                action = "SELL"
                quantity = shares_before

            # Apply the trade if any
            trade_pnl_abs = None
            trade_pnl_pct = None
            holding_period_days = None
            tx_id = None

            if action == "BUY" and quantity > 0:
                cost = quantity * execution_price
                cash -= cost
                # update average cost
                total_cost = pos.cost_basis * pos.shares + cost
                pos.shares += quantity
                pos.cost_basis = total_cost / pos.shares if pos.shares > 0 else 0
                pos.entry_date = d.strftime("%Y-%m-%d")
                pos.entry_price = execution_price
                pos.last_signal_id = sig_id
                tx_id = f"T-{pid}-{_short_uuid()}"
            elif action == "SELL" and quantity > 0:
                proceeds = quantity * execution_price
                cash += proceeds
                # realised pnl
                trade_pnl_abs = (execution_price - pos.cost_basis) * quantity
                trade_pnl_pct = (execution_price / pos.cost_basis - 1) if pos.cost_basis > 0 else None
                if pos.entry_date:
                    try:
                        ed = pd.to_datetime(pos.entry_date)
                        holding_period_days = int((d - ed).days)
                    except Exception:
                        holding_period_days = None
                trade_pnls.append(float(trade_pnl_abs) if trade_pnl_abs is not None else 0.0)
                pos.shares = 0
                pos.cost_basis = 0
                pos.entry_date = None
                pos.entry_price = None
                tx_id = f"T-{pid}-{_short_uuid()}"

            # post-trade snapshot
            shares_after = pos.shares
            pos_val_after = shares_after * execution_price
            portfolio_value_after = cash + sum(
                positions[tt].shares * _price_on(prices_by_ticker[tt], d, "Close")
                for tt in tickers if positions[tt].shares > 0
            )
            weight_actual_after = pos_val_after / portfolio_value_after if portfolio_value_after > 0 else 0

            # forward returns of the underlying (not portfolio) — these are
            # really for *signal* effectiveness analysis, not for P&L.
            r1, r5, r20 = _forward_returns(df, d)

            decision_record = _build_decision_record(
                portfolio_cfg=portfolio_cfg, ticker=t, day=d,
                model_name=sig_model, sig_row=sig_row, sig_id=sig_id,
                decision_id=decision_id, action=action,
                execution_price=execution_price, prev_close=prev_close,
                row=row, quantity=quantity,
                shares_before=shares_before, shares_after=shares_after,
                cash_before=cash_before, cash_after=cash,
                pos_val_before=pos_val_before, pos_val_after=pos_val_after,
                portfolio_value_before=portfolio_value_before,
                portfolio_value_after=portfolio_value_after,
                weight_target=target_weight, weight_actual=weight_actual_after,
                weight_before=weight_actual_before,
                trade_pnl_abs=trade_pnl_abs, trade_pnl_pct=trade_pnl_pct,
                holding_period_days=holding_period_days,
                reason=reason, transaction_id=tx_id,
                generated_at=generated_at,
                data_version=data_version, strategy_version=strategy_version,
                label_mode=label_mode, label_version=label_version,
                r1=r1, r5=r5, r20=r20,
            )
            decisions.append(decision_record)
            if tx_id is not None:
                transactions.append(decision_record)  # transaction = decision with tx_id set

        # End of day equity curve point
        ev = cash + sum(
            positions[t].shares * _price_on(prices_by_ticker[t], d, "Close")
            for t in tickers if positions[t].shares > 0
        )
        equity_curve.append({
            "date": d.strftime("%Y-%m-%d"),
            "portfolio_value": round(float(ev), 2),
            "cash": round(float(cash), 2),
            "invested": round(float(ev - cash), 2),
        })

    # Compute drawdown and PnL columns on top of decisions
    _annotate_drawdown(equity_curve)
    _annotate_drawdown_and_cum_pnl(transactions)

    # Final positions (after last day)
    last_date = all_dates[-1]
    positions_list = []
    for t, pos in positions.items():
        if pos.shares <= 0:
            continue
        last_price = _price_on(prices_by_ticker[t], last_date, "Close")
        positions_list.append({
            "ticker": t,
            "shares": round(float(pos.shares), 4),
            "avg_cost": round(float(pos.cost_basis), 4),
            "last_price": round(float(last_price), 4),
            "market_value": round(float(pos.shares * last_price), 2),
            "unrealised_pnl_abs": round(float((last_price - pos.cost_basis) * pos.shares), 2),
            "unrealised_pnl_pct": round(float(last_price / pos.cost_basis - 1), 6) if pos.cost_basis > 0 else None,
            "entry_date": pos.entry_date,
            "entry_price": pos.entry_price,
            "last_signal_id": pos.last_signal_id,
        })

    # Portfolio performance metrics
    eq_series = pd.Series([p["portfolio_value"] for p in equity_curve],
                          index=pd.to_datetime([p["date"] for p in equity_curve]))
    perf = metrics_mod.portfolio_metrics(eq_series)
    trade = metrics_mod.trade_metrics(trade_pnls)

    bench_perf = {}
    if benchmark_prices is not None and not benchmark_prices.empty:
        bench_eq = (benchmark_prices["Close"].reindex(eq_series.index).ffill()
                    / benchmark_prices["Close"].reindex(eq_series.index).ffill().iloc[0]) * initial_capital
        bench_perf = metrics_mod.portfolio_metrics(bench_eq.dropna())

    final_value = equity_curve[-1]["portfolio_value"] if equity_curve else initial_capital

    metadata = {
        "generated_at_utc": generated_at,
        "data_version": data_version,
        "strategy_version": strategy_version,
        "signal_source": signal_source,
        "rules": rules,
        "n_decisions": len(decisions),
        "n_transactions": len(transactions),
        "first_date": all_dates[0].strftime("%Y-%m-%d") if all_dates else None,
        "last_date": all_dates[-1].strftime("%Y-%m-%d") if all_dates else None,
        "label_mode": label_mode,
        "label_version": label_version,
    }

    return BacktestResult(
        portfolio_id=pid,
        portfolio_name=portfolio_cfg["name"],
        description=portfolio_cfg.get("description", ""),
        initial_capital=initial_capital,
        final_value=final_value,
        cash=float(cash),
        transactions=transactions,
        decisions=decisions,
        equity_curve=equity_curve,
        positions=positions_list,
        metrics={**perf, **{"trade_" + k: v for k, v in trade.items()}},
        benchmark_metrics=bench_perf,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def math_floor(x: float) -> float:
    """Floor positive shares to whole units. Returns 0 if x < 1."""
    if x < 1:
        return 0.0
    return float(int(x))


def _price_on(df: pd.DataFrame, day, col: str = "Close") -> float:
    if day in df.index:
        v = df.loc[day, col]
        if pd.notna(v):
            return float(v)
    # forward fill from nearest earlier date
    before = df.loc[:day]
    if before.empty:
        return 0.0
    val = before[col].iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def _previous_close(df: pd.DataFrame, day) -> Optional[float]:
    before = df.loc[:day]
    if len(before) < 2:
        return None
    val = before["Close"].iloc[-2]
    return float(val) if pd.notna(val) else None


def _forward_returns(df: pd.DataFrame, day):
    """Return realised 1d / 5d / 20d returns *after* ``day``. NaN if not enough future."""
    if day not in df.index:
        return None, None, None
    idx = df.index.get_loc(day)
    closes = df["Close"]
    base = closes.iloc[idx]
    out = []
    for h in (1, 5, 20):
        if idx + h < len(closes):
            future = closes.iloc[idx + h]
            if pd.notna(future) and pd.notna(base) and base > 0:
                out.append(round(float(future / base - 1), 6))
            else:
                out.append(None)
        else:
            out.append(None)
    return out[0], out[1], out[2]


def _resolve_signal_for_day(signal_source: str, ticker: str, day,
                            signals_by_model: Dict[str, pd.DataFrame],
                            rules: Dict) -> Optional[Dict]:
    """Pick the signal row that should drive the portfolio decision today."""
    if signal_source == "buy_and_hold":
        df = signals_by_model.get("buy_and_hold")
    elif signal_source == "ensemble_majority":
        df = signals_by_model.get("ensemble_majority")
    else:
        df = signals_by_model.get(signal_source)
    if df is None or day not in df.index:
        return None
    row = df.loc[day]
    sig_id = row.get("signal_id") or f"S-{ticker}-{signal_source}-{day.strftime('%Y%m%d')}"
    return {"row": row.to_dict(), "model_name": signal_source, "signal_id": sig_id}


def _build_decision_record(portfolio_cfg, ticker, day, model_name, sig_row, sig_id,
                           decision_id, action, execution_price, prev_close,
                           row, quantity, shares_before, shares_after,
                           cash_before, cash_after, pos_val_before, pos_val_after,
                           portfolio_value_before, portfolio_value_after,
                           weight_target, weight_actual, weight_before,
                           trade_pnl_abs, trade_pnl_pct, holding_period_days,
                           reason, transaction_id, generated_at,
                           data_version, strategy_version,
                           label_mode, label_version,
                           r1, r5, r20):
    """Assemble the rich decision/transaction record (matches the prompt's spec)."""
    def _f(x, n=4):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return None
        try:
            return round(float(x), n)
        except (TypeError, ValueError):
            return None

    return {
        "portfolio_id": portfolio_cfg["id"],
        "portfolio_name": portfolio_cfg["name"],
        "ticker": ticker,
        "model_name": model_name,
        "signal_source": model_name,
        "strategy_name": portfolio_cfg.get("signal_source", "technical_rule_based"),
        "label_mode": label_mode,
        "label_version": label_version,

        "decision_id": decision_id,
        "signal_id": sig_id,
        "transaction_id": transaction_id,

        "action": action,
        "decision_timestamp_utc": generated_at,
        "market_date": day.strftime("%Y-%m-%d"),
        "execution_date": day.strftime("%Y-%m-%d"),
        "execution_price": _f(execution_price, 4),
        "previous_close": _f(prev_close, 4),

        "open": _f(row.get("Open"), 4),
        "high": _f(row.get("High"), 4),
        "low": _f(row.get("Low"), 4),
        "close": _f(row.get("Close"), 4),
        "volume": int(row["Volume"]) if pd.notna(row.get("Volume")) else None,

        "quantity": _f(quantity, 4),
        "shares_before": _f(shares_before, 4),
        "shares_after": _f(shares_after, 4),

        "cash_before": _f(cash_before, 2),
        "cash_after": _f(cash_after, 2),
        "position_value_before": _f(pos_val_before, 2),
        "position_value_after": _f(pos_val_after, 2),
        "portfolio_value_before": _f(portfolio_value_before, 2),
        "portfolio_value_after": _f(portfolio_value_after, 2),

        "weight_target": _f(weight_target, 4),
        "weight_actual_before": _f(weight_before, 4),
        "weight_actual": _f(weight_actual, 4),

        "score": _f(sig_row.get("score"), 4),
        "confidence": _f(sig_row.get("confidence"), 4),
        "probability_up": _f(sig_row.get("probability_up"), 4),
        "threshold_used": _f(sig_row.get("threshold_used"), 4),

        "rsi": _f(row.get("RSI"), 2),
        "macd": _f(row.get("MACD"), 4),
        "macd_signal": _f(row.get("MACD_signal"), 4),
        "macd_hist": _f(row.get("MACD_hist"), 4),
        "bb_upper": _f(row.get("BB_upper"), 4),
        "bb_lower": _f(row.get("BB_lower"), 4),
        "bb_middle": _f(row.get("BB_middle"), 4),
        "sma20": _f(row.get("SMA20"), 4),
        "sma50": _f(row.get("SMA50"), 4),

        "return_1d_after": r1,
        "return_5d_after": r5,
        "return_20d_after": r20,

        "trade_pnl_abs": _f(trade_pnl_abs, 2),
        "trade_pnl_pct": _f(trade_pnl_pct, 6),
        "holding_period_days": holding_period_days,
        "hit": (trade_pnl_abs is not None and trade_pnl_abs > 0) if trade_pnl_abs is not None else None,

        "reason": reason,
        "triggered_rules": sig_row.get("triggered_rules", ""),

        "data_version": data_version,
        "strategy_version": strategy_version,
        "generated_at_utc": generated_at,
    }


def _annotate_drawdown(equity_curve: List[Dict]) -> None:
    """Annotate equity_curve in place with running max / drawdown."""
    peak = -float("inf")
    for point in equity_curve:
        v = point["portfolio_value"]
        peak = max(peak, v)
        dd = (v - peak) / peak if peak > 0 else 0
        point["peak"] = round(peak, 2)
        point["drawdown"] = round(dd, 6)


def _annotate_drawdown_and_cum_pnl(transactions: List[Dict]) -> None:
    cum = 0.0
    max_dd = 0.0
    peak = -float("inf")
    for tx in transactions:
        pnl = tx.get("trade_pnl_abs") or 0.0
        cum += pnl
        peak = max(peak, cum)
        dd = (cum - peak) / abs(peak) if peak != 0 else 0.0
        max_dd = min(max_dd, dd)
        tx["cumulative_pnl"] = round(cum, 2)
        tx["drawdown_after_trade"] = round(dd, 6)
        tx["max_drawdown_to_date"] = round(max_dd, 6)
