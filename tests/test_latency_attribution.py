"""Known-answer tests for latency_attribution.py."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest_core as bc  # noqa: E402
import latency_attribution as la  # noqa: E402

BASE = dict(min_accepted=0.25, std_factor=0.0, max_positions=1, cost_bps=0.0, min_price=0.0)


def frames(data):
    out = {}
    for t, cols in data.items():
        n = len(cols["Close"])
        df = pd.DataFrame({k: v for k, v in cols.items() if k != "prob"}, index=pd.bdate_range("2022-01-03", periods=n))
        df.index.name = "Date"
        for h in bc.PERIODS:
            df[f"Pred_Prob_{h}"] = cols.get("prob", [0.0] * n) if h == "1d" else 0.0
            df[f"Pred_Prob_Std_{h}"] = 0.0
        out[t] = df
    return out


def test_raw_name():
    assert la.raw_name("AAL_daily_processed") == "AAL_daily.csv"
    assert la.raw_name("A") == "A_daily.csv"


def test_attach_open_rejects_mismatched_bars(tmp_path):
    f = frames({"AAA_daily_processed": {"Close": [10.0, 11.0, 12.0]}})
    pd.DataFrame({"date": pd.bdate_range("2022-01-03", periods=3), "open": [9.5, 10.5, 11.5],
                  "high": 0, "low": 0, "close": [10.0, 22.0, 12.01], "volume": 0}).to_csv(
        tmp_path / "AAA_daily.csv", index=False)
    out, rep = la.attach_open(f, tmp_path)
    o = out["AAA_daily_processed"]["Open"].tolist()
    assert o[0] == 9.5 and np.isnan(o[1]) and o[2] == 11.5     # day 2 close 22 vs 11 -> different bar
    assert rep["rows_close_mismatch"] == 1 and rep["missing_raw_files"] == []
    out2, rep2 = la.attach_open({"ZZZ_daily_processed": f["AAA_daily_processed"]}, tmp_path)
    assert rep2["missing_raw_files"] == ["ZZZ_daily_processed"] and out2["ZZZ_daily_processed"]["Open"].isna().all()


def _panel():
    # AAA signals on day 0. Close 10 -> open 11 -> close 12 (1d hold).
    return bc.build_panel(frames({
        "AAA": {"Open": [10, 11, 13, 13], "Close": [10, 12, 13, 13], "prob": [0.9, 0, 0, 0]},
        "BBB": {"Open": [20, 20, 20, 20], "Close": [20, 22, 22, 22]},
        "CCC": {"Open": [30, 30, 30, 30], "Close": [30, 30, 30, 30]},
    }))


def test_decomposition_sums_to_trade_return_and_replays():
    pn = _panel()
    a = bc.run_strategy(pn, bc.StrategyParams(**BASE, fill="same_close")).trades
    dec = la.decompose_trades(a, pn)
    r = dec.iloc[0]
    assert r["on"] == pytest.approx(math.log(11 / 10)) and r["s1"] == pytest.approx(math.log(12 / 11))
    assert r["rest"] == pytest.approx(0.0)                       # 1d trade exits at close t+1
    assert r["on"] + r["s1"] + r["rest"] == pytest.approx(r["total"])
    assert r["total"] == pytest.approx(math.log(1.2))


def test_candidates_match_engine_choice():
    pn = _panel()
    cand, days = la.candidate_signals(pn, bc.StrategyParams(**BASE))
    assert cand.sum() == 1 and cand[0, pn.tickers.index("AAA")] and days[0, pn.tickers.index("AAA")] == 1
    tr = bc.run_strategy(pn, bc.StrategyParams(**BASE)).trades
    assert list(zip(tr["Ticker"], tr["SignalDate"])) == [("AAA", pn.dates[0])]


def test_event_excess_hand_computed():
    pn = _panel()
    ev = la.event_table(pn, bc.StrategyParams(**BASE))
    assert len(ev) == 1
    e = ev.iloc[0]
    g = [math.log(12 / 10), math.log(22 / 20), 0.0]                 # universe gaps on day 0
    assert e["gap"] == pytest.approx(g[0])
    assert e["x_gap"] == pytest.approx(g[0] - np.mean(g))
    on = [math.log(11 / 10), 0.0, 0.0]
    assert e["x_on"] == pytest.approx(on[0] - np.mean(on))
    post = [math.log(13 / 12), 0.0, 0.0]                            # 1-session hold after close t+1
    assert e["x_post"] == pytest.approx(post[0] - np.mean(post))
    assert e["UniverseN"] == 3 and not e["Flagged"]


def test_split_flag_excludes_from_primary():
    pn = bc.build_panel(frames({
        "AAA": {"Open": [10, 5.2, 5, 5], "Close": [10, 5.1, 5, 5], "prob": [0.9, 0, 0, 0]},   # 2:1 split overnight
        "BBB": {"Open": [20] * 4, "Close": [20] * 4},
    }))
    p = bc.StrategyParams(**{**BASE, "spike_threshold": 10.0})
    assert la.event_table(pn, p).empty
    allev = la.event_table(pn, p, exclude_flagged=False)
    assert len(allev) == 1 and bool(allev.iloc[0]["Flagged"])


def test_daily_mean_stat_weights_days_equally():
    d = pd.to_datetime(["2022-01-03", "2022-01-03", "2022-01-03", "2022-01-04"])
    ev = pd.DataFrame({"Date": d, "x": [0.03, 0.03, 0.03, -0.01]})
    s = la.daily_mean_stat(ev, "x")
    assert s["mean"] == pytest.approx(0.01) and s["event_mean"] == pytest.approx(0.02)
    assert s["n_days"] == 2 and s["n_events"] == 4


def test_gate_g1_logic():
    good = {"mean": 0.003, "nw_t": 2.5}
    years = {y: 0.001 for y in range(2016, 2024)}
    assert la.gate_g1(good, years)["pass"]
    assert not la.gate_g1({"mean": 0.0019, "nw_t": 5.0}, years)["pass"]
    assert not la.gate_g1({"mean": 0.01, "nw_t": 1.9}, years)["pass"]
    four = {y: (0.001 if y < 2020 else -0.001) for y in range(2016, 2024)}
    assert not la.gate_g1(good, four)["pass"]


def test_match_trades():
    a = pd.DataFrame({"Ticker": ["X", "Y"], "SignalDate": pd.to_datetime(["2022-01-03", "2022-01-04"])})
    c = pd.DataFrame({"Ticker": ["X", "Z"], "SignalDate": pd.to_datetime(["2022-01-03", "2022-01-05"])})
    assert la.match_trades(a, c) == {"matched": 1, "only_first": 1, "only_second": 1}
