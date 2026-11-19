"""Unit tests for backtest_core: known-answer, no-lookahead, controls."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest_core as bc  # noqa: E402


def make_forecasts(prices: dict[str, list[float]], probs: dict[str, list[float]], std: float = 0.0):
    dates = pd.bdate_range("2022-01-03", periods=len(next(iter(prices.values()))))
    out = {}
    for t, px in prices.items():
        df = pd.DataFrame({"Close": px}, index=dates)
        df.index.name = "Date"
        for h in bc.PERIODS:
            # only the 1d horizon carries signal; others below threshold
            df[f"Pred_Prob_{h}"] = probs[t] if h == "1d" else 0.0
            df[f"Pred_Prob_Std_{h}"] = std
        out[t] = df
    return out


BASE = dict(min_accepted=0.25, std_factor=0.0, max_positions=1, cost_bps=0.0, min_price=0.0)


def test_next_bar_fill_known_answer():
    # Signal only on day 0; price path 10, 11, 12, 13. Next-bar: buy day1 @11, 1d hold -> sell day2 @12.
    f = make_forecasts({"AAA": [10, 11, 12, 13]}, {"AAA": [0.9, 0.0, 0.0, 0.0]})
    res = bc.run_strategy(bc.build_panel(f), bc.StrategyParams(**BASE, next_bar_fill=True))
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["EntryPrice"] == 11 and tr["ExitPrice"] == 12
    assert res.equity[-1] == pytest.approx(5000 * 12 / 11)


def test_same_bar_fill_reproduces_old_behaviour():
    f = make_forecasts({"AAA": [10, 11, 12, 13]}, {"AAA": [0.9, 0.0, 0.0, 0.0]})
    res = bc.run_strategy(bc.build_panel(f), bc.StrategyParams(**BASE, next_bar_fill=False))
    tr = res.trades.iloc[0]
    assert tr["EntryPrice"] == 10 and tr["ExitPrice"] == 11


def test_costs_charged_both_sides():
    f = make_forecasts({"AAA": [10, 10, 10, 10]}, {"AAA": [0.9, 0.0, 0.0, 0.0]})
    p = bc.StrategyParams(**{**BASE, "cost_bps": 100.0})
    res = bc.run_strategy(bc.build_panel(f), p)
    # flat price, 1% in and 1% out: 5000 -> 4950 shares value -> minus 1% = 4900.5
    assert res.equity[-1] == pytest.approx(5000 * 0.99 * 0.99)
    assert res.total_costs == pytest.approx(50 + 49.5)


def test_no_lookahead_future_prices_do_not_change_past_decisions():
    px = [10, 11, 12, 13, 14, 15]
    pr = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0]
    a = bc.run_strategy(bc.build_panel(make_forecasts({"AAA": px}, {"AAA": pr})), bc.StrategyParams(**BASE))
    px2 = px[:3] + [1000, 0.5, 99]  # change only prices after the trade closed
    b = bc.run_strategy(bc.build_panel(make_forecasts({"AAA": px2}, {"AAA": pr})), bc.StrategyParams(**BASE))
    pd.testing.assert_frame_equal(a.trades, b.trades)
    np.testing.assert_allclose(a.equity[:3], b.equity[:3])


def test_causal_spike_filter_only_blocks_after_spike():
    # BBB spikes +100% on day 3. Its signal on day 1 (before spike) must still trade under causal mode.
    f = make_forecasts(
        {"BBB": [10, 10, 10, 20, 20, 20]},
        {"BBB": [0.0, 0.9, 0.0, 0.0, 0.9, 0.0]},
    )
    causal = bc.run_strategy(bc.build_panel(f), bc.StrategyParams(**BASE, spike_mode="causal"))
    full = bc.run_strategy(bc.build_panel(f), bc.StrategyParams(**BASE, spike_mode="full"))
    assert list(causal.trades["SignalDate"].dt.day) == [4]          # day-1 signal (Jan 4) only
    assert len(full.trades) == 0                                    # look-ahead version drops the ticker entirely


def test_ranking_picks_highest_adjusted_prob_and_std_penalty():
    f = make_forecasts({"AAA": [10] * 4, "BBB": [10] * 4}, {"AAA": [0.6, 0, 0, 0], "BBB": [0.5, 0, 0, 0]})
    # give AAA a large std so the penalty flips the ranking
    f["AAA"]["Pred_Prob_Std_1d"] = 0.3
    res = bc.run_strategy(bc.build_panel(f), bc.StrategyParams(**{**BASE, "std_factor": 1.0}))
    assert res.trades.iloc[0]["Ticker"] == "BBB"


def test_shuffle_with_single_ticker_is_identity():
    f = make_forecasts({"AAA": [10, 11, 12, 13]}, {"AAA": [0.9, 0.0, 0.0, 0.0]})
    panel = bc.build_panel(f)
    a = bc.run_strategy(panel, bc.StrategyParams(**BASE))
    b = bc.run_strategy(panel, bc.StrategyParams(**BASE), rng=np.random.default_rng(0), shuffle_signals=True)
    np.testing.assert_allclose(a.equity, b.equity)


def test_deterministic_given_seed():
    rng_prices = np.random.default_rng(1).lognormal(0, 0.02, size=(60, 5)).cumprod(axis=0) * 20
    rng_probs = np.random.default_rng(2).uniform(0, 0.6, size=(60, 5))
    f = make_forecasts({f"T{k}": list(rng_prices[:, k]) for k in range(5)},
                       {f"T{k}": list(rng_probs[:, k]) for k in range(5)})
    panel = bc.build_panel(f)
    p = bc.StrategyParams(**{**BASE, "max_positions": 3})
    a = bc.run_strategy(panel, p, rng=np.random.default_rng(7), shuffle_signals=True)
    b = bc.run_strategy(panel, p, rng=np.random.default_rng(7), shuffle_signals=True)
    np.testing.assert_array_equal(a.equity, b.equity)


def test_newey_west_matches_plain_t_for_white_noise():
    r = np.random.default_rng(3).normal(0.001, 0.01, 5000)
    plain = r.mean() / (r.std(ddof=0) / np.sqrt(r.size))
    assert bc.newey_west_t(r, lags=10) == pytest.approx(plain, rel=0.1)


def test_non_common_symbols():
    assert bc.is_non_common("AIIOW") and bc.is_non_common("AGNCP") and bc.is_non_common("AACBU")
    assert not bc.is_non_common("AAPL") and not bc.is_non_common("AMZN")
