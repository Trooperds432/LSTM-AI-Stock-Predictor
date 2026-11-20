"""Known-answer tests for TrainingData/pit/costs.py (spread estimators, ADV, volatility)."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from TrainingData.pit import costs  # noqa: E402

IDX2 = pd.bdate_range("2022-01-03", periods=2)


def s(vals, idx=None):
    idx = idx if idx is not None else pd.bdate_range("2022-01-03", periods=len(vals))
    return pd.Series(vals, index=idx, dtype=float)


# ---------------------------------------------------------------- Corwin-Schultz

def test_cs_equal_ranges_closed_form():
    # Same range ln(H/L)=x on both days and the same H and L: beta=2x^2, gamma=x^2 => alpha = x exactly,
    # so S = 2(e^x - 1)/(1 + e^x). With H=110, L=100: e^x = 1.1 -> S = 0.2/2.1 = 2/21.
    h, l, c = s([110, 110]), s([100, 100]), s([105, 105])
    out = costs.corwin_schultz_daily(h, l, c)
    assert np.isnan(out.iloc[0])                       # dated on the second day of the pair
    assert out.iloc[1] == pytest.approx(2 / 21, rel=1e-12)


def test_cs_negative_estimate_set_to_zero_without_overnight_adjustment():
    # Narrow ranges with a big gap between days -> gamma dominates -> alpha < 0 -> 0
    h, l, c = s([101, 121]), s([99, 119]), s([100, 120])
    assert costs.corwin_schultz_daily(h, l, c, overnight_adjust=False).iloc[1] == 0.0


def test_cs_overnight_adjustment_hand_computed():
    # Day 2's whole range (119..121) is above day 1's close (100): shift it down by 119-100=19 -> 100..102.
    h, l, c = s([101, 121]), s([99, 119]), s([100, 120])
    beta = math.log(101 / 99) ** 2 + math.log(102 / 100) ** 2
    gamma = math.log(102 / 99) ** 2
    k = 3 - 2 * math.sqrt(2)
    alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / k - math.sqrt(gamma / k)
    expected = max(2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha)), 0.0)
    assert costs.corwin_schultz_daily(h, l, c).iloc[1] == pytest.approx(expected, rel=1e-12)


def test_cs_overnight_adjustment_gap_down():
    # Day 2 entirely below day 1's close: shift up by C0 - H1 = 100 - 81 = 19 -> 98..100... (79..81 -> 98..100)
    h, l, c = s([101, 81]), s([99, 79]), s([100, 80])
    shifted = costs.corwin_schultz_daily(s([101, 100]), s([99, 98]), s([100, 99]), overnight_adjust=False)
    assert costs.corwin_schultz_daily(h, l, c).iloc[1] == pytest.approx(shifted.iloc[1], rel=1e-12)


# ---------------------------------------------------------------- Abdi-Ranaldo

def test_ar_hand_computed():
    # logs: day1 lnH=.01 lnL=-.01 (eta=0) lnC=.01; day2 lnH=0 lnL=-.02 (eta=-.01)
    # s^2 = 4 (0.01 - 0)(0.01 + 0.01) = 0.0008
    h = s(np.exp([0.01, 0.0])); l = s(np.exp([-0.01, -0.02])); c = s(np.exp([0.01, -0.01]))
    out = costs.abdi_ranaldo_daily(h, l, c)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(math.sqrt(0.0008), rel=1e-9)


def test_ar_negative_product_set_to_zero():
    # day2 eta = 0.02 > c1 = 0.01 -> 4 (0.01)(0.01 - 0.02) < 0 -> 0
    h = s(np.exp([0.01, 0.03])); l = s(np.exp([-0.01, 0.01])); c = s(np.exp([0.01, 0.02]))
    assert costs.abdi_ranaldo_daily(h, l, c).iloc[1] == 0.0


# ---------------------------------------------------------------- rolling median / wide frames

def test_rolling_median_21_and_wide_frames():
    n = 30
    idx = pd.bdate_range("2022-01-03", periods=n)
    # ticker A: constant range 110/100 every day -> every daily estimate is 2/21
    hi = pd.DataFrame({"A": 110.0, "B": 110.0}, index=idx)
    lo = pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)
    cl = pd.DataFrame({"A": 105.0, "B": 105.0}, index=idx)
    hi.loc[idx[25], "B"] = np.nan                     # a missing bar only removes two pair estimates
    out = costs.corwin_schultz_spread(hi, lo, cl)
    assert out["A"].iloc[:21].isna().all()            # first estimate on day 1, 21 needed -> first value day 21
    assert out["A"].iloc[21] == pytest.approx(2 / 21)
    assert out["B"].iloc[21] == pytest.approx(2 / 21)
    assert out["B"].iloc[26:].isna().all()            # only 19-20 valid estimates in those windows
    lenient = costs.corwin_schultz_spread(hi, lo, cl, min_periods=15)
    assert lenient["B"].iloc[29] == pytest.approx(2 / 21)


def test_rolling_median_is_a_median():
    x = s([1.0] * 10 + [100.0] * 2 + [3.0] * 9)       # 21 values, median 1? sorted: 10 ones, 9 threes, 2 hundreds -> 11th = 3
    assert costs.rolling_median(x).iloc[-1] == 3.0


# ---------------------------------------------------------------- ADV / volatility

def test_adv20_known_answer():
    v = s(np.arange(1, 26))
    a = costs.adv(v)
    assert np.isnan(a.iloc[18])
    assert a.iloc[19] == pytest.approx(10.5)          # mean(1..20)
    assert a.iloc[24] == pytest.approx(15.5)          # mean(6..25)


def test_adv20_restates_volume_across_split():
    v = s([100.0] * 20 + [200.0])                     # 2-for-1 split on day 20 doubles share volume
    f = s([1.0] * 20 + [2.0])
    assert costs.adv(v, split_factor=f).iloc[20] == pytest.approx(200.0)
    assert costs.adv(v).iloc[20] == pytest.approx(105.0)


def test_dollar_adv():
    assert costs.dollar_adv(s([10.0] * 20), s([100.0] * 20)).iloc[19] == pytest.approx(1000.0)


def test_realized_vol_split_adjusted():
    px = 100 * 1.01 ** np.arange(30)
    px[15:] /= 2                                      # raw prices halve on the split session
    f = s([1.0] * 15 + [2.0] + [1.0] * 14)
    adj = costs.realized_vol(s(px), split_factor=f)
    raw = costs.realized_vol(s(px))
    assert adj.iloc[-1] == pytest.approx(0.0, abs=1e-12)   # constant 1% log-ish returns -> zero std
    assert adj.iloc[20] == pytest.approx(0.0, abs=1e-12)
    assert raw.iloc[20] > 0.1
