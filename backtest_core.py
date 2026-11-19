"""Pure, testable backtest engine for the LSTM probability strategy.

Design rules (see research/audit_2026-11-19_preregistration.md):
- Signals use only information up to and including day t.
- With next_bar_fill=True an order decided on day t fills at day t+1's close.
- The spike filter is causal: a ticker is ineligible on day t once any daily
  move above the threshold has occurred at or before t.
- Costs are charged on every buy and sell, in basis points of notional.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PERIODS: dict[str, int] = {"1d": 1, "1w": 5, "1m": 21, "6m": 126}


@dataclass(frozen=True)
class StrategyParams:
    min_accepted: float = 0.25
    std_factor: float = 1.0
    max_positions: int = 5
    cost_bps: float = 20.0
    min_price: float = 5.0
    spike_threshold: float = 0.80
    spike_mode: str = "causal"      # "causal" (point-in-time) or "full" (old look-ahead behaviour)
    next_bar_fill: bool = True      # False reproduces the old same-close fill
    initial_value: float = 5000.0


@dataclass
class Panel:
    """Wide arrays aligned on dates (rows) x tickers (columns)."""
    dates: list[pd.Timestamp]
    tickers: list[str]
    close: np.ndarray                 # [T, N], NaN where no bar
    prob: dict[str, np.ndarray]       # horizon -> [T, N]
    std: dict[str, np.ndarray]        # horizon -> [T, N]
    max_move_to_date: np.ndarray      # [T, N] expanding max |daily pct change| (causal)
    max_move_full: np.ndarray         # [N] full-period max |daily pct change| (look-ahead, attribution only)


@dataclass
class Result:
    equity: np.ndarray
    trades: pd.DataFrame
    buy_points: list[dict] = field(default_factory=list)
    total_costs: float = 0.0
    open_at_end: int = 0
    days_invested: int = 0
    traded_notional: float = 0.0


def is_non_common(symbol: str) -> bool:
    """5+ letter symbols ending in W/U/R/P are warrants, units, rights or preferred shares."""
    return len(symbol) >= 5 and symbol[-1] in "WURP"


def load_forecasts(forecast_dir: Path, exclude_non_common: bool = True) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load *_forecast.csv files. Returns (ticker -> DataFrame indexed by Date, excluded symbols)."""
    cols = ["Date", "Close"] + [f"Pred_Prob_{h}" for h in PERIODS] + [f"Pred_Prob_Std_{h}" for h in PERIODS]
    out: dict[str, pd.DataFrame] = {}
    excluded: list[str] = []
    for filename in sorted(os.listdir(forecast_dir)):
        if not filename.endswith("_forecast.csv"):
            continue
        ticker = filename.split("_forecast")[0]
        symbol = ticker.split("_")[0]
        if exclude_non_common and is_non_common(symbol):
            excluded.append(symbol)
            continue
        df = pd.read_csv(Path(forecast_dir) / filename, parse_dates=["Date"], usecols=lambda c: c in cols)
        out[ticker] = df.set_index("Date").sort_index()
    return out, excluded


def build_panel(forecasts: dict[str, pd.DataFrame], start: str | pd.Timestamp | None = None) -> Panel:
    tickers = sorted(forecasts)
    all_dates = sorted(set().union(*[set(df.index) for df in forecasts.values()]))
    if start is not None:
        all_dates = [d for d in all_dates if d >= pd.Timestamp(start)]
    idx = pd.DatetimeIndex(all_dates)

    def wide(col: str) -> np.ndarray:
        return np.column_stack([
            forecasts[t][col].reindex(idx).to_numpy(dtype=float) if col in forecasts[t] else np.full(len(idx), np.nan)
            for t in tickers
        ]) if tickers else np.empty((len(idx), 0))

    close = wide("Close")
    prob = {h: wide(f"Pred_Prob_{h}") for h in PERIODS}
    std = {h: wide(f"Pred_Prob_Std_{h}") for h in PERIODS}

    # Daily moves on each ticker's own bar sequence (not the union calendar)
    max_to_date = np.zeros_like(close)
    max_full = np.zeros(len(tickers))
    for j, t in enumerate(tickers):
        c = forecasts[t]["Close"]
        c = c[c.index >= idx[0]] if len(idx) else c
        moves = c.pct_change().abs().fillna(0.0)
        max_full[j] = float(moves.max()) if len(moves) else 0.0
        max_to_date[:, j] = moves.cummax().reindex(idx).ffill().fillna(0.0).to_numpy()

    return Panel(list(idx), tickers, close, prob, std, max_to_date, max_full)


def _eligible_mask(panel: Panel, i: int, p: StrategyParams) -> np.ndarray:
    c = panel.close[i]
    ok = np.isfinite(c) & (c >= p.min_price)
    if p.spike_mode == "causal":
        ok &= panel.max_move_to_date[i] <= p.spike_threshold
    elif p.spike_mode == "full":
        ok &= panel.max_move_full <= p.spike_threshold
    else:
        raise ValueError(f"unknown spike_mode {p.spike_mode!r}")
    return ok


def run_strategy(panel: Panel, p: StrategyParams, rng: np.random.Generator | None = None,
                 shuffle_signals: bool = False) -> Result:
    """Simulate the strategy. With shuffle_signals=True (negative control) each day's
    (prob, std) pairs are randomly reassigned among that day's tickers that have a forecast."""
    T, N = panel.close.shape
    cost_rate = p.cost_bps / 10_000.0
    horizons = list(PERIODS.items())

    cash = p.initial_value
    positions: list[dict] = []
    pending: list[dict] = []
    equity = np.zeros(T)
    trades: list[dict] = []
    buy_points: list[dict] = []
    total_costs = 0.0
    days_invested = 0
    traded = 0.0
    last_px = np.full(N, np.nan)

    for i in range(T):
        row = panel.close[i]
        fin = np.isfinite(row)
        last_px[fin] = row[fin]

        # 1) Mark to market
        for pos in positions:
            pos["last_px"] = last_px[pos["j"]]

        # 2) Exits
        keep = []
        for pos in positions:
            if i < pos["exit_idx"]:
                keep.append(pos)
                continue
            gross = pos["shares"] * pos["last_px"]
            cost = gross * cost_rate
            cash += gross - cost
            total_costs += cost
            traded += gross
            net_ret = (gross - cost) / pos["cost_basis"] - 1.0
            trades.append({
                "BuyDate": panel.dates[pos["entry_idx"]], "SellDate": panel.dates[i],
                "SignalDate": panel.dates[pos["signal_idx"]],
                "Ticker": panel.tickers[pos["j"]], "Horizon": pos["h"], "DaysHeld": i - pos["entry_idx"],
                "Pred_Prob": pos["prob"], "Pred_Prob_Std": pos["std"], "Adj_Prob": pos["adj"],
                "EntryPrice": pos["entry_px"], "ExitPrice": pos["last_px"],
                "Gross_Return%": (pos["last_px"] / pos["entry_px"] - 1.0) * 100,
                "Actual_Return%": net_ret * 100,
                "Actual_LogR": np.log1p(net_ret),
                "PnL$": gross - cost - pos["cost_basis"],
            })
        positions = keep

        def equity_now() -> float:
            return cash + sum(q["shares"] * q["last_px"] for q in positions)

        # 3) Fill pending orders from yesterday's signals at today's close
        if pending:
            slot = equity_now() / p.max_positions
            for o in pending:
                px = row[o["j"]]
                if not np.isfinite(px) or px <= 0 or any(q["j"] == o["j"] for q in positions):
                    continue
                alloc = min(slot, cash)
                if alloc < 1.0:
                    break
                cost = alloc * cost_rate
                cash -= alloc
                total_costs += cost
                traded += alloc
                positions.append({**o, "entry_idx": i, "entry_px": px, "last_px": px,
                                  "shares": (alloc - cost) / px, "cost_basis": alloc,
                                  "exit_idx": min(i + o["days"], T - 1)})
                buy_points.append({"Date": panel.dates[i], "Value": equity_now(), "Ticker": panel.tickers[o["j"]],
                                   "Horizon": o["h"], "Pred_Prob": o["prob"], "Adj_Prob": o["adj"]})
            pending = []

        # 4) New signals (decided at today's close)
        free = p.max_positions - len(positions)
        # No new signals on the final day: an order could never be held
        if free > 0 and i <= T - 2:
            elig = _eligible_mask(panel, i, p)
            for q in positions:
                elig[q["j"]] = False

            probs = {h: panel.prob[h][i].copy() for h, _ in horizons}
            stds = {h: panel.std[h][i].copy() for h, _ in horizons}
            if shuffle_signals:
                assert rng is not None, "shuffle_signals needs an rng"
                has = np.isfinite(probs[horizons[0][0]])
                cols = np.flatnonzero(has)
                perm = rng.permutation(cols)
                for h, _ in horizons:
                    probs[h][cols] = probs[h][perm]
                    stds[h][cols] = stds[h][perm]

            best_adj = np.full(N, -np.inf)
            best_h = np.empty(N, dtype=object)
            for h, _ in horizons:
                s = np.where(np.isfinite(stds[h]), stds[h], 0.0)
                adj = probs[h] - p.std_factor * np.maximum(s, 1e-6)
                better = np.isfinite(adj) & (adj > best_adj)
                best_adj[better] = adj[better]
                best_h[better] = h
            cand = np.flatnonzero(elig & (best_adj > p.min_accepted))
            cand = cand[np.argsort(-best_adj[cand], kind="stable")][:free]

            orders = []
            for j in cand:
                h = best_h[j]
                orders.append({"j": int(j), "h": h, "days": PERIODS[h], "signal_idx": i,
                               "prob": float(panel.prob[h][i, j]) if not shuffle_signals else float(probs[h][j]),
                               "std": float(stds[h][j]) if np.isfinite(stds[h][j]) else 0.0,
                               "adj": float(best_adj[j])})
            if p.next_bar_fill:
                pending = orders
            else:
                # Old behaviour: fill at the same close the signal was computed on
                slot = equity_now() / p.max_positions
                for o in orders:
                    px = row[o["j"]]
                    alloc = min(slot, cash)
                    if alloc < 1.0:
                        break
                    cost = alloc * cost_rate
                    cash -= alloc
                    total_costs += cost
                    traded += alloc
                    positions.append({**o, "entry_idx": i, "entry_px": px, "last_px": px,
                                      "shares": (alloc - cost) / px, "cost_basis": alloc,
                                      "exit_idx": min(i + o["days"], T - 1)})
                    buy_points.append({"Date": panel.dates[i], "Value": equity_now(), "Ticker": panel.tickers[o["j"]],
                                       "Horizon": o["h"], "Pred_Prob": o["prob"], "Adj_Prob": o["adj"]})

        if positions:
            days_invested += 1
        equity[i] = cash + sum(q["shares"] * q["last_px"] for q in positions)

    return Result(equity, pd.DataFrame(trades), buy_points, total_costs, len(positions), days_invested, traded)


# ----------------------------------------------------------------------------- metrics

def daily_returns(equity: np.ndarray) -> np.ndarray:
    e = np.asarray(equity, dtype=float)
    r = e[1:] / e[:-1] - 1.0
    return r[np.isfinite(r)]


def max_drawdown(equity: np.ndarray) -> float:
    e = np.asarray(equity, dtype=float)
    if len(e) < 2:
        return 0.0
    return float((e / np.maximum.accumulate(e) - 1.0).min())


def sharpe(r: np.ndarray, periods: int = 252) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1) if r.size > 1 else 0.0
    return float(np.sqrt(periods) * r.mean() / sd) if sd > 1e-12 else float("nan")


def sortino(r: np.ndarray, periods: int = 252) -> float:
    r = np.asarray(r, dtype=float)
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) if r.size else 0.0
    return float(np.sqrt(periods) * r.mean() / downside) if downside > 1e-12 else float("nan")


def newey_west_t(r: np.ndarray, lags: int = 10) -> float:
    """t-stat of the mean with Newey-West (Bartlett) HAC standard error."""
    r = np.asarray(r, dtype=float)
    n = r.size
    if n < lags + 2:
        return float("nan")
    u = r - r.mean()
    var = u @ u / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1)
        var += 2.0 * w * (u[k:] @ u[:-k]) / n
    se = np.sqrt(var / n)
    return float(r.mean() / se) if se > 0 else float("nan")


def summarize(res: Result, initial_value: float) -> dict:
    r = daily_returns(res.equity)
    t = res.trades
    pnl = t["PnL$"] if len(t) else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    avg_equity = float(np.mean(res.equity)) if len(res.equity) else initial_value
    years = max(len(res.equity) - 1, 1) / 252.0
    return {
        "total_return": float(res.equity[-1] / initial_value - 1.0),
        "cagr": float((res.equity[-1] / initial_value) ** (1 / years) - 1.0) if res.equity[-1] > 0 else float("nan"),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_drawdown": max_drawdown(res.equity),
        "nw_t_stat": newey_west_t(r),
        "trades": int(len(t)),
        "win_rate": float((pnl > 0).mean()) if len(t) else float("nan"),
        "avg_win_pct": float(t.loc[pnl > 0, "Actual_Return%"].mean()) if len(wins) else float("nan"),
        "avg_loss_pct": float(t.loc[pnl <= 0, "Actual_Return%"].mean()) if len(losses) else float("nan"),
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("nan"),
        "time_in_market": res.days_invested / max(len(res.equity), 1),
        "annual_turnover": float(res.traded_notional / avg_equity / years),
        "total_costs": float(res.total_costs),
        "open_at_end": res.open_at_end,
    }
