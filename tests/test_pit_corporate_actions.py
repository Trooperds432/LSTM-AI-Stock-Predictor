"""Tests for pit.corporate_actions, pit.spreads and the consolidated-RIC mapping."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TrainingData"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TrainingData" / "pit"))
from pit import corporate_actions as ca, spreads  # noqa: E402
from lseg_consolidated import consolidated_ric  # noqa: E402


def panel(closes, opens=None, isin="X", start="2020-01-02"):
    s = pd.bdate_range(start, periods=len(closes)).strftime("%Y-%m-%d")
    opens = opens or closes
    return pd.DataFrame({"session": s, "isin": isin, "open": opens, "close": closes})


def ev(factor, eff, typ="Share Split", isin="X"):
    return pd.DataFrame([{"Instrument": "X.N", "Corporate Change Event Type": typ, "Capital Change Effective Date": eff,
                          "Adjustment Factor": factor, "isin": isin}])


def test_split_verified_on_next_session_when_lseg_date_is_day_before():
    px = panel([100, 100, 50, 50], opens=[100, 100, 50, 50])          # halves at 2020-01-06
    out = ca.verify_ratio_events(ev(0.5, "2020-01-03"), px)
    assert out.status.iloc[0] == "verified" and out.trading_session.iloc[0] == "2020-01-06"


def test_combined_reverse_split_and_spinoff_verified_together():
    px = panel([10, 10, 14, 14], opens=[10, 10, 14, 14])               # 2x reverse * 0.7 spin = 1.4
    e = pd.concat([ev(2.0, "2020-01-06", "Share Consolidation"), ev(0.7, "2020-01-06", "Demerger")], ignore_index=True)
    out = ca.verify_ratio_events(e, px)
    assert set(out.status) == {"verified_combined"} and set(out.trading_session) == {"2020-01-06"}


def test_factors_preserve_value_across_split_and_dividend():
    px = panel([100, 100, 50, 50])
    events = pd.DataFrame([{"isin": "X", "trading_session": "2020-01-06", "factor": 0.5, "status": "verified"}])
    divs = pd.DataFrame([{"isin": "X", "ex_date": "2020-01-07", "amount": 5.0}])
    f = ca.build_factors(px, events, divs).set_index("session")
    adj = px.set_index("session").close * f.split_factor
    assert list(adj.round(6)) == [50, 50, 50, 50]                       # split-adjusted series is flat
    assert f.tr_factor["2020-01-03"] == pytest.approx(0.5 * 0.9)       # 5 / 50 dividend on 01-07
    assert f.tr_factor["2020-01-07"] == 1.0


def test_spread_estimate_ignores_splits():
    n = 40
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    h, l = c * 1.01, c * 0.99
    raw = pd.DataFrame({"session": pd.bdate_range("2020-01-02", periods=n).strftime("%Y-%m-%d"), "isin": "X",
                        "high": h, "low": l, "close": c})
    split = raw.copy()
    split.loc[20:, ["high", "low", "close"]] /= 2                       # 2-for-1 split at row 20
    fac = pd.DataFrame({"session": raw.session, "isin": "X", "split_factor": [0.5] * 20 + [1.0] * 20})
    a = spreads.estimate(raw, fac.assign(split_factor=1.0)).spread_est
    b = spreads.estimate(split, fac).spread_est
    np.testing.assert_allclose(a.values[:20] * 0.5 / 0.5, b.values[:20], rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(a.values[25:], b.values[25:], rtol=1e-9, equal_nan=True)


@pytest.mark.parametrize("ric,expected", [("AAPL.OQ", "AAPL.O"), ("JPM.N", "JPM"), ("CEPH.OQ^J11", "CEPH.O^J11"),
                                          ("GR.N^G12", "GR^G12"), ("BRKb.N", "BRKb")])
def test_consolidated_ric_mapping(ric, expected):
    assert consolidated_ric(ric) == expected
