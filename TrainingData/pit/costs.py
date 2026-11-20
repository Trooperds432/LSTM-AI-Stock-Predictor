"""Execution-cost inputs from daily bars: bid-ask spread estimators, ADV and volatility.

Pure functions. Every input is a pandas Series (one ticker, indexed by session) or a wide
DataFrame (sessions x tickers); outputs have the same shape and index.

Point-in-time convention: a value dated session t uses only bars at or before t. The two-day
spread estimators use the pair (t-1, t), so they are dated t (NaN on the first session).

Spreads are *fractional full* spreads (0.01 = 1% of price); half of it is the cost of one fill.

References
- Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low
  Prices", Journal of Finance 67(2).
- Abdi & Ranaldo (2017), "A Simple Estimation of Bid-Ask Spreads from Daily Close, High, and Low
  Prices", Review of Financial Studies 30(12).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPREAD_WINDOW = 21
ADV_WINDOW = 20
VOL_WINDOW = 20

_K = 3.0 - 2.0 * np.sqrt(2.0)   # Corwin-Schultz constant


def corwin_schultz_daily(high, low, close=None, overnight_adjust: bool = True):
    """Two-day Corwin-Schultz spread for each pair (t-1, t), dated t.

    With overnight_adjust (paper section II.B, needs close) day t's high and low are shifted
    by the overnight gap when day t's whole range lies above or below the close of t-1.
    Negative two-day estimates are set to 0, as in the paper.
    """
    h1, l1 = high, low
    h0, l0 = high.shift(1), low.shift(1)
    if overnight_adjust:
        if close is None:
            raise ValueError("overnight_adjust needs close")
        c0 = close.shift(1)
        gap = (l1 - c0).where(l1 > c0, 0.0)
        gap = gap.where(~(h1 < c0), h1 - c0)
        gap = gap.where(c0.notna(), 0.0)
        h1, l1 = h1 - gap, l1 - gap
    beta = np.log(h0 / l0) ** 2 + np.log(h1 / l1) ** 2
    hmax = np.maximum(h0, h1)
    lmin = np.minimum(l0, l1)
    gamma = np.log(hmax / lmin) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return s.clip(lower=0.0)


def abdi_ranaldo_daily(high, low, close):
    """Two-day Abdi-Ranaldo spread for each pair (t-1, t), dated t.

    s^2 = 4 (c_{t-1} - eta_{t-1}) (c_{t-1} - eta_t), with c the log close and eta the log
    mid-range (log H + log L) / 2. Negative s^2 is set to 0 before the square root.
    """
    eta = (np.log(high) + np.log(low)) / 2.0
    c = np.log(close)
    s2 = 4.0 * (c.shift(1) - eta.shift(1)) * (c.shift(1) - eta)
    return np.sqrt(s2.clip(lower=0.0))


def rolling_median(x, window: int = SPREAD_WINDOW, min_periods: int | None = None):
    """Trailing median over `window` sessions (inclusive of t). NaNs are skipped."""
    mp = window if min_periods is None else min_periods
    return x.rolling(window, min_periods=mp).median()


def corwin_schultz_spread(high, low, close, window: int = SPREAD_WINDOW, min_periods: int | None = None,
                          overnight_adjust: bool = True):
    """21-session trailing median of the two-day Corwin-Schultz estimates (dated t)."""
    return rolling_median(corwin_schultz_daily(high, low, close, overnight_adjust), window, min_periods)


def abdi_ranaldo_spread(high, low, close, window: int = SPREAD_WINDOW, min_periods: int | None = None):
    """21-session trailing median of the two-day Abdi-Ranaldo estimates (dated t).

    The paper averages s^2 over the month before the square root; a median of daily estimates
    is used here (plan Phase 5) because it is robust to single-day outliers.
    """
    return rolling_median(abdi_ranaldo_daily(high, low, close), window, min_periods)


def _cum_split(split_factor, like):
    """Cumulative product of split factors to date (1.0 where none). split_factor[t] = new shares
    per old share, effective at session t."""
    if split_factor is None:
        return None
    f = split_factor.reindex_like(like) if hasattr(split_factor, "reindex_like") else split_factor
    return f.fillna(1.0).cumprod()


def adv(volume, window: int = ADV_WINDOW, split_factor=None, min_periods: int | None = None):
    """Average daily share volume over the trailing `window` sessions, in session-t share units.

    With split_factor, earlier volumes are restated into the share count in force at t, so a
    2-for-1 split does not halve the measured liquidity.
    """
    mp = window if min_periods is None else min_periods
    cum = _cum_split(split_factor, volume)
    if cum is None:
        return volume.rolling(window, min_periods=mp).mean()
    return (volume / cum).rolling(window, min_periods=mp).mean() * cum


def dollar_adv(close, volume, window: int = ADV_WINDOW, min_periods: int | None = None):
    """Average daily dollar volume (close x volume) over the trailing window. Split-invariant."""
    mp = window if min_periods is None else min_periods
    return (close * volume).rolling(window, min_periods=mp).mean()


def realized_vol(close, window: int = VOL_WINDOW, split_factor=None, min_periods: int | None = None):
    """Trailing standard deviation (ddof=1) of daily log close-to-close returns.

    With split_factor the return across a split session t is log(C_t * f_t / C_{t-1}).
    """
    mp = window if min_periods is None else min_periods
    r = np.log(close / close.shift(1))
    if split_factor is not None:
        f = split_factor.reindex_like(close) if hasattr(split_factor, "reindex_like") else split_factor
        r = r + np.log(f.fillna(1.0))
    return r.rolling(window, min_periods=mp).std(ddof=1)
