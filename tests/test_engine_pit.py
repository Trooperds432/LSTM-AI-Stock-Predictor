"""Known-answer tests for the point-in-time extensions of backtest_core: fills, universe mask,
corporate actions, settlement, cost model, participation cap and the confirmation guard."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest_core as bc  # noqa: E402

START = "2022-01-03"
BASE = dict(min_accepted=0.25, std_factor=0.0, max_positions=1, cost_bps=0.0, min_price=0.0)


def frames(data: dict[str, dict[str, list]], start: str = START) -> dict[str, pd.DataFrame]:
    """data: ticker -> {"Close": [...], "Open": [...], "prob": [...], ...}. Only the 1d horizon carries signal."""
    out = {}
    for t, cols in data.items():
        n = len(cols["Close"])
        df = pd.DataFrame({k: v for k, v in cols.items() if k != "prob"},
                          index=pd.bdate_range(start, periods=n))
        df.index.name = "Date"
        prob = cols.get("prob", [0.0] * n)
        for h in bc.PERIODS:
            df[f"Pred_Prob_{h}"] = prob if h == "1d" else 0.0
            df[f"Pred_Prob_Std_{h}"] = 0.0
        out[t] = df
    return out


def day(k: int) -> pd.Timestamp:
    return pd.bdate_range(START, periods=k + 1)[-1]


# ---------------------------------------------------------------- fills

def test_next_open_fill_known_answer():
    f = frames({"AAA": {"Open": [10, 11, 12, 13], "Close": [10.5, 11.5, 12.5, 13.5], "prob": [0.9, 0, 0, 0]}})
    res = bc.run_strategy(bc.build_panel(f), bc.StrategyParams(**BASE, fill="next_open"))
    tr = res.trades.iloc[0]
    assert tr["EntryPrice"] == 11 and tr["ExitPrice"] == 12      # buy open(t+1), 1d hold -> sell open(t+2)
    assert tr["BuyDate"] == day(1) and tr["SellDate"] == day(2)
    assert res.equity[1] == pytest.approx(5000 / 11 * 11.5)      # marked at the close after an open fill
    assert res.equity[-1] == pytest.approx(5000 * 12 / 11)


def test_fill_alias_matches_legacy_flag():
    f = frames({"AAA": {"Close": [10, 11, 12, 13], "prob": [0.9, 0, 0, 0]}})
    pn = bc.build_panel(f)
    legacy = bc.run_strategy(pn, bc.StrategyParams(**BASE, next_bar_fill=True))
    named = bc.run_strategy(pn, bc.StrategyParams(**BASE, fill="next_close"))
    pd.testing.assert_frame_equal(legacy.trades, named.trades)
    old_same = bc.run_strategy(pn, bc.StrategyParams(**BASE, next_bar_fill=False))
    new_same = bc.run_strategy(pn, bc.StrategyParams(**BASE, fill="same_close"))
    pd.testing.assert_frame_equal(old_same.trades, new_same.trades)
    assert new_same.trades.iloc[0]["EntryPrice"] == 10


def test_fill_validation():
    pn = bc.build_panel(frames({"AAA": {"Close": [10, 11, 12], "prob": [0.9, 0, 0]}}))
    with pytest.raises(ValueError, match="Open"):
        bc.run_strategy(pn, bc.StrategyParams(**BASE, fill="next_open"))
    with pytest.raises(ValueError, match="unknown fill"):
        bc.run_strategy(pn, bc.StrategyParams(**BASE, fill="vwap"))
    with pytest.raises(ValueError, match="contradicts"):
        bc.run_strategy(pn, bc.StrategyParams(**BASE, fill="next_open", next_bar_fill=False))


def test_next_open_sizing_does_not_peek_at_todays_close():
    # AAA bought at day-1 open; BBB's order fills at day-2 open. Changing AAA's day-2 CLOSE must not
    # change BBB's size (only prices known at the open may size an open fill).
    def run(aaa_close_day2):
        f = frames({
            "AAA": {"Open": [10, 10, 10, 10, 10], "Close": [10, 10, aaa_close_day2, 10, 10], "prob": [0.9, 0, 0, 0, 0]},
            "BBB": {"Open": [10, 10, 10, 10, 10], "Close": [10] * 5, "prob": [0, 0.8, 0, 0, 0]},
        })
        p = bc.StrategyParams(**{**BASE, "max_positions": 2}, fill="next_open")
        tr = bc.run_strategy(bc.build_panel(f), p).trades
        return tr.loc[tr["Ticker"] == "BBB", "PnL$"].iloc[0], tr.loc[tr["Ticker"] == "BBB", "BuyDate"].iloc[0]

    assert run(10.0) == run(30.0)


def test_next_open_exit_deferred_through_halt():
    f = frames({"AAA": {"Open": [10, 11, np.nan, 14, 15], "Close": [10, 11, 12, 13, 14], "prob": [0.9, 0, 0, 0, 0]}})
    tr = bc.run_strategy(bc.build_panel(f), bc.StrategyParams(**BASE, fill="next_open")).trades.iloc[0]
    assert tr["SellDate"] == day(3) and tr["ExitPrice"] == 14


# ---------------------------------------------------------------- universe mask

def test_member_mask_blocks_entry():
    f = frames({
        "AAA": {"Close": [10] * 4, "Member": [False, True, True, True], "prob": [0.9, 0, 0, 0]},
        "BBB": {"Close": [10] * 4, "Member": [True] * 4, "prob": [0.5, 0, 0, 0]},
    })
    pn = bc.build_panel(f)
    masked = bc.run_strategy(pn, bc.StrategyParams(**BASE, use_member_mask=True))
    unmasked = bc.run_strategy(pn, bc.StrategyParams(**BASE))
    assert list(masked.trades["Ticker"]) == ["BBB"]
    assert list(unmasked.trades["Ticker"]) == ["AAA"]


def test_member_mask_requires_member_column():
    pn = bc.build_panel(frames({"AAA": {"Close": [10] * 3, "prob": [0.9, 0, 0]}}))
    with pytest.raises(ValueError, match="Member"):
        bc.run_strategy(pn, bc.StrategyParams(**BASE, use_member_mask=True))


# ---------------------------------------------------------------- corporate actions

def test_split_preserves_value():
    # Buy 250 sh at day-1 close 20; 2-for-1 effective day 2 (raw close 10) -> 500 sh x 10 = 5000.
    f = frames({"AAA": {"Close": [20, 20, 10, 10], "prob": [0.9, 0, 0, 0]}})
    splits = pd.DataFrame({"date": [day(2)], "ticker": ["AAA"], "factor": [2.0]})
    pn = bc.build_panel(f, splits=splits)
    res = bc.run_strategy(pn, bc.StrategyParams(**BASE))
    tr = res.trades.iloc[0]
    assert res.equity[2] == pytest.approx(5000.0)
    assert tr["EntryPrice"] == pytest.approx(10.0) and tr["ExitPrice"] == 10
    assert tr["Gross_Return%"] == pytest.approx(0.0) and tr["PnL$"] == pytest.approx(0.0)
    # the raw -50% print is not a "spike" once the split is known
    assert pn.max_move_to_date[:, 0].max() == pytest.approx(0.0)
    # without the split input the same raw prices lose half the money
    assert bc.run_strategy(bc.build_panel(f), bc.StrategyParams(**BASE)).equity[2] == pytest.approx(2500.0)


def test_split_on_non_session_maps_to_next_session():
    f = frames({"AAA": {"Close": [20] * 5 + [10] * 2}})
    saturday = pd.Timestamp("2022-01-08")              # sessions: Mon 3 .. Fri 7, Mon 10 (index 5)
    pn = bc.build_panel(f, splits=pd.DataFrame({"date": [saturday], "ticker": ["AAA"], "factor": [2.0]}))
    assert pn.dates[5] == pd.Timestamp("2022-01-10")
    assert pn.split_factor[5, 0] == 2.0 and pn.split_factor[:, 0].prod() == 2.0
    assert pn.max_move_to_date[:, 0].max() == pytest.approx(0.0)


def test_dividend_credited_on_ex_date():
    # 250 sh bought at day-1 close 20; $0.50 ex-date day 2, close drops to 19.5 -> 250*19.5 + 125 = 5000
    f = frames({"AAA": {"Close": [20, 20, 19.5, 19.5], "prob": [0.9, 0, 0, 0]}})
    divs = pd.DataFrame({"ex_date": [day(2)], "ticker": ["AAA"], "cash": [0.5]})
    res = bc.run_strategy(bc.build_panel(f, dividends=divs), bc.StrategyParams(**BASE))
    tr = res.trades.iloc[0]
    assert res.dividends == pytest.approx(125.0)
    assert res.equity[2] == pytest.approx(5000.0)
    assert tr["Dividends$"] == pytest.approx(125.0) and tr["PnL$"] == pytest.approx(0.0)


def test_dividend_not_credited_when_bought_on_ex_date():
    # Signal day 1 -> bought at day-2 close (the ex-date): not a holder of record
    f = frames({"AAA": {"Close": [20, 20, 19.5, 19.5], "prob": [0, 0.9, 0, 0]}})
    divs = pd.DataFrame({"ex_date": [day(2)], "ticker": ["AAA"], "cash": [0.5]})
    res = bc.run_strategy(bc.build_panel(f, dividends=divs), bc.StrategyParams(**BASE))
    assert res.dividends == 0.0 and res.trades.iloc[0]["BuyDate"] == day(2)


def _merger_panel(cash, succ_shares, bbb_open_day3=60.0):
    # AAA: bought day-1 close at 20 (250 sh), no bars from day 3; settlement paid on day 3.
    nan = np.nan
    f = frames({
        "AAA": {"Open": [20, 20, 20, nan, nan, nan], "Close": [20, 20, 20, nan, nan, nan],
                "prob": [0.9, 0, 0, 0, 0, 0]},
        "BBB": {"Open": [50, 50, 50, bbb_open_day3, 61, 62], "Close": [50, 50, 50, 59, 61, 62], "prob": [0.0] * 6},
    })
    # hold horizon 1w so the time exit would come after the settlement
    for h in bc.PERIODS:
        f["AAA"][f"Pred_Prob_{h}"] = [0.9, 0, 0, 0, 0, 0] if h == "1w" else 0.0
    st = pd.DataFrame({"date": [day(3)], "ticker": ["AAA"], "cash_per_share": [cash],
                       "successor_ticker": ["BBB" if succ_shares else None], "successor_shares": [succ_shares]})
    return bc.build_panel(f, settlements=st)


@pytest.mark.parametrize("cash,ss,value_ps", [(25.0, 0.0, 25.0),      # cash merger / cash delisting
                                               (0.0, 0.5, 30.0),      # stock: 0.5 x BBB open 60
                                               (10.0, 0.5, 40.0)])    # mixed: 10 + 0.5 x 60
def test_settlement_known_values(cash, ss, value_ps):
    res = bc.run_strategy(_merger_panel(cash, ss), bc.StrategyParams(**BASE))
    tr = res.trades.iloc[0]
    assert tr["ExitType"] == "settlement" and tr["SellDate"] == day(3)
    assert tr["ExitPrice"] == pytest.approx(value_ps)
    assert res.equity[-1] == pytest.approx(250 * value_ps)
    assert res.open_at_end == 0


def test_stock_settlement_waits_for_first_successor_open():
    # BBB has no open on day 3; first successor session is day 4 (open 61). Day 3 is marked at BBB's last close.
    res = bc.run_strategy(_merger_panel(0.0, 0.5, bbb_open_day3=np.nan), bc.StrategyParams(**BASE))
    tr = res.trades.iloc[0]
    assert tr["SellDate"] == day(4) and tr["ExitPrice"] == pytest.approx(30.5)
    assert res.equity[3] == pytest.approx(250 * 0.5 * 59)
    assert res.equity[-1] == pytest.approx(250 * 30.5)


def test_settled_ticker_cannot_be_entered():
    nan = np.nan
    f = frames({"AAA": {"Open": [20, 20, 20, 20], "Close": [20, 20, 20, 20], "prob": [0, 0, 0, 0.9]},
                "BBB": {"Open": [5] * 4, "Close": [5] * 4}})
    f["AAA"]["Pred_Prob_1d"] = [0, 0.9, 0.9, 0]
    st = pd.DataFrame({"date": [day(1)], "ticker": ["AAA"], "cash_per_share": [20.0],
                       "successor_ticker": [nan], "successor_shares": [nan]})
    assert len(bc.run_strategy(bc.build_panel(f, settlements=st), bc.StrategyParams(**BASE)).trades) == 0


def test_settlement_successor_must_be_in_panel():
    f = frames({"AAA": {"Open": [20, 20], "Close": [20, 20]}})
    st = pd.DataFrame({"date": [day(1)], "ticker": ["AAA"], "cash_per_share": [0.0],
                       "successor_ticker": ["ZZZ"], "successor_shares": [1.0]})
    with pytest.raises(ValueError, match="successor"):
        bc.build_panel(f, settlements=st)


# ---------------------------------------------------------------- costs

FEES = (("2020-01-01", 27.80e-6, 0.000166, 8.30),)


def test_fees_only_on_sells():
    # 500 sh at $10 flat. Buy: no fee. Sell: SEC 5000 x 27.80e-6 = 0.139; TAF 500 x 0.000166 = 0.083
    f = frames({"AAA": {"Close": [10] * 4, "prob": [0.9, 0, 0, 0]}})
    res = bc.run_strategy(bc.build_panel(f), bc.StrategyParams(**BASE, sell_fee_table=FEES))
    assert res.cost_breakdown["fees"] == pytest.approx(0.139 + 0.083)
    assert res.total_costs == pytest.approx(0.222)
    assert res.equity[-1] == pytest.approx(5000 - 0.222)


def test_taf_cap_and_dated_table():
    # 1,000,000 sh: TAF capped at 8.30. Later row in force from 2022-01-04 (day 1) applies to the day-2 sale.
    f = frames({"AAA": {"Close": [0.01] * 4, "prob": [0.9, 0, 0, 0]}})
    table = (("2020-01-01", 0.0, 0.000166, 8.30), ("2022-01-04", 0.0, 0.000200, 9.00))
    p = bc.StrategyParams(**BASE, initial_value=10_000.0, sell_fee_table=table)
    res = bc.run_strategy(bc.build_panel(f), p)
    assert res.cost_breakdown["fees"] == pytest.approx(9.00)
    with pytest.raises(ValueError, match="no rate"):
        bc.run_strategy(bc.build_panel(f), bc.StrategyParams(**BASE, sell_fee_table=(("2030-01-01", 0, 0, 0),)))


def _fee_on(date, notional=1e6, shares=1000.0):
    return bc.sell_fees(pd.Timestamp(date), notional, shares, bc._fee_rows(bc.load_sell_fee_table()))


@pytest.mark.parametrize("date,sec_per_million,taf,cap", [
    ("2011-06-30", 19.20, 0.000075, 3.75),
    ("2012-07-02", 22.40, 0.000119, 5.95),
    ("2015-06-01", 18.40, 0.000119, 5.95),
    ("2021-03-01", 5.10, 0.000119, 5.95),
    ("2022-05-13", 5.10, 0.000130, 6.49),     # day before the FY2022 SEC change
    ("2022-05-16", 22.90, 0.000130, 6.49),
    ("2024-06-03", 27.80, 0.000166, 8.30),
    ("2025-06-02", 0.00, 0.000166, 8.30),     # SEC $0 from 2025-05-14; 2025 TAF unchanged
    ("2026-05-01", 20.60, 0.000195, 9.79),
    ("2026-10-15", 20.60, 0.0, 0.0),          # FINRA TAF pause Oct-Dec 2026
])
def test_fee_rate_table_values(date, sec_per_million, taf, cap):
    # $1m notional, 1,000 shares (TAF below cap) and 1,000,000 shares (TAF at cap)
    assert _fee_on(date) == pytest.approx(sec_per_million + min(taf * 1000, cap))
    assert _fee_on(date, shares=1e6) == pytest.approx(sec_per_million + min(taf * 1e6, cap))


def test_fee_rate_table_refuses_unknown_dates():
    with pytest.raises(ValueError, match="unknown"):
        _fee_on("2027-02-01")                 # 2027 TAF unresolved between two FINRA filings
    with pytest.raises(ValueError, match="no rate"):
        _fee_on("2008-06-02")                 # before the sourced history


def test_fee_rate_csv_is_sourced():
    df = pd.read_csv(bc.FEE_RATES_CSV, parse_dates=["effective_date"])
    assert set(df["fee"]) == {"sec31", "finra_taf"}
    assert df["source_url"].str.match(r"https://www\.(sec\.gov|finra\.org)/").all()
    for _, g in df.groupby("fee"):
        assert g["effective_date"].is_monotonic_increasing and not g["effective_date"].duplicated().any()
    known = df.dropna(subset=["rate"])
    assert (known["rate"] >= 0).all()
    assert known.loc[known["fee"] == "finra_taf", "max_per_trade"].notna().all()


def test_half_spread_cost_both_sides():
    # 1% full spread: buy pays 25 -> 497.5 sh; sell 4975 pays 24.875 -> 4950.125
    f = frames({"AAA": {"Close": [10] * 4, "SpreadEst": [0.01] * 4, "prob": [0.9, 0, 0, 0]}})
    res = bc.run_strategy(bc.build_panel(f), bc.StrategyParams(**BASE))
    assert res.equity[-1] == pytest.approx(4950.125)
    assert res.cost_breakdown["spread"] == pytest.approx(25 + 24.875)
    off = bc.run_strategy(bc.build_panel(f), bc.StrategyParams(**BASE, use_spread_cost=False))
    assert off.equity[-1] == pytest.approx(5000.0)


def test_spread_uses_prior_session_estimate_and_counts_fallback():
    # Day-0 estimate missing; the day-1 buy uses it -> fallback 0.2%; the day-2 sell uses day-1's 1%.
    f = frames({"AAA": {"Close": [10] * 4, "SpreadEst": [np.nan, 0.01, 0.5, 0.5], "prob": [0.9, 0, 0, 0]}})
    res = bc.run_strategy(bc.build_panel(f), bc.StrategyParams(**BASE))
    assert res.spread_fallbacks == 1
    buy = 0.5 * 0.002 * 5000
    assert res.cost_breakdown["spread"] == pytest.approx(buy + 0.5 * 0.01 * (5000 - buy))


def test_square_root_impact_known_answer():
    f = frames({"AAA": {"Close": [10] * 4, "Volume": [1e6] * 4, "prob": [0.9, 0, 0, 0]}})
    pn = bc.build_panel(f)
    pn.adv20[:] = 10_000.0          # known inputs instead of 20 sessions of history
    pn.sigma20[:] = 0.02
    p = bc.StrategyParams(**BASE, impact_k=1.0, participation_cap=None)
    res = bc.run_strategy(pn, p)
    buy = 0.02 * np.sqrt(500 / 10_000) * 5000                    # 500 sh pre-cost
    sh = (5000 - buy) / 10
    sell = 0.02 * np.sqrt(sh / 10_000) * sh * 10
    assert res.cost_breakdown["impact"] == pytest.approx(buy + sell)
    assert res.equity[-1] == pytest.approx(sh * 10 - sell)


def test_participation_cap_shrinks_order():
    n = 25
    probs = [0.0] * n
    probs[21] = 0.9                              # fill on day 22 uses ADV20 through day 21
    f = frames({"AAA": {"Close": [10.0] * n, "Volume": [1000.0] * n, "prob": probs}})
    res = bc.run_strategy(bc.build_panel(f), bc.StrategyParams(**BASE))   # default cap 5% of ADV20
    tr = res.trades.iloc[0]
    assert res.cap_shrunk == 1
    assert tr["PnL$"] == pytest.approx(0.0)
    assert res.traded_notional == pytest.approx(2 * 50 * 10)  # 50 sh = 5% x 1000, bought and sold
    assert res.equity[-1] == pytest.approx(5000.0)


def test_participation_cap_blocks_entry_without_adv_history():
    f = frames({"AAA": {"Close": [10.0] * 4, "Volume": [1000.0] * 4, "prob": [0.9, 0, 0, 0]}})
    res = bc.run_strategy(bc.build_panel(f), bc.StrategyParams(**BASE))
    assert len(res.trades) == 0 and res.cap_blocked == 1
    nocap = bc.run_strategy(bc.build_panel(f), bc.StrategyParams(**BASE, participation_cap=None))
    assert len(nocap.trades) == 1


def test_pit_preset():
    p = bc.StrategyParams.pit()
    assert p.fill_mode == "next_open" and p.use_member_mask and p.cost_bps == 0.0 and p.impact_k == 1.0
    assert bc.StrategyParams().fill_mode == "next_close"      # legacy default unchanged


# ---------------------------------------------------------------- confirmation-period guard

def test_guard_constant_matches_data_layer():
    from TrainingData.pit import config
    assert bc.CONFIRM_START == config.CONFIRM_START == pd.Timestamp("2025-01-02")


def test_confirmation_guard_raises():
    f = frames({"AAA": {"Close": [10] * 4, "prob": [0.9, 0, 0, 0]}}, start="2024-12-30")  # ... 2025-01-02
    with pytest.raises(bc.ConfirmationPeriodError):
        bc.build_panel(f)
    pn = bc.build_panel(f, confirm=True)
    with pytest.raises(bc.ConfirmationPeriodError):
        bc.run_strategy(pn, bc.StrategyParams(**BASE))
    assert len(bc.run_strategy(pn, bc.StrategyParams(**BASE), confirm=True).trades) == 1
    # research runs can cut the panel before the reserved period
    cut = bc.build_panel(f, end="2025-01-01")          # bdate_range fixture includes 1 Jan
    assert cut.dates[-1] == pd.Timestamp("2025-01-01")
    bc.run_strategy(cut, bc.StrategyParams(**BASE))


def test_guard_allows_last_research_day():
    f = frames({"AAA": {"Close": [10] * 3}}, start="2024-12-27")   # 27, 30, 31 Dec 2024
    assert bc.build_panel(f).dates[-1] == pd.Timestamp("2024-12-31")
