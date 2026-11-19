"""
mosaic_score.py
===============

Daily, lag-safe Mosaic Theory scoring for equity ML pipelines.

Purpose
-------
This module converts public, point-in-time feature blocks into a compact
set of daily Mosaic Theory scores that can be merged into an LSTM or hybrid
sequence/context model.

It is designed to sit *after* feature builders such as:
- insider_features.py
- macro_cross_asset.py
- event_calendar.py

The scoring framework follows the requested insider-weighted structure:
- Insiders:            35%
- Technicals:          20%
- Fundamentals:        20%
- Institutional/Flows: 15%
- Macro/Cross-asset:   10%

Key outputs
-----------
- mosaic_insiders_score
- mosaic_technicals_score
- mosaic_fundamentals_score
- mosaic_flows_score
- mosaic_macro_score
- mosaic_total_score               (0-100)
- mosaic_score_raw                 (-10 to +10)
- mosaic_vol_adjusted_score
- mosaic_confidence                (0-100)
- mosaic_rating                    (Bearish/Neutral/Bullish)
- mosaic_pass_bullish_threshold
- mosaic_supporting_tiles_count

Notes
-----
- Public-data only.
- This module does not fetch data from the internet by itself.
- It assumes all inputs were built in a lag-safe, point-in-time manner.
- Missing features degrade confidence rather than causing hard failure.
- If a feature is unavailable, the relevant bucket remains closer to neutral.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def _to_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)



def _series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")



def _bool_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        s = df[col]
        if s.dtype == bool:
            return s.astype(float)
        if pd.api.types.is_numeric_dtype(s):
            return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(float)
        return s.astype(str).str.lower().isin({"1", "true", "yes", "y"}).astype(float)
    return pd.Series(default, index=df.index, dtype="float64")



def _clip01(x: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(np.clip(np.asarray(x, dtype=float), 0.0, 1.0))



def _clip100(x: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(np.clip(np.asarray(x, dtype=float), 0.0, 100.0))



def _bounded_tanh(x: pd.Series | np.ndarray, scale: float = 1.0) -> pd.Series:
    arr = np.asarray(x, dtype=float) / (scale if scale else 1.0)
    return pd.Series(np.tanh(arr))



def _safe_div(numer: pd.Series, denom: pd.Series | float, fill: float = 0.0) -> pd.Series:
    denom_arr = np.asarray(denom, dtype=float)
    out = np.divide(
        np.asarray(numer, dtype=float),
        denom_arr,
        out=np.full_like(np.asarray(numer, dtype=float), fill, dtype=float),
        where=np.isfinite(denom_arr) & (denom_arr != 0),
    )
    return pd.Series(out, index=numer.index)



def _rolling_zscore(s: pd.Series, window: int = 60, min_periods: int = 20) -> pd.Series:
    mean = s.rolling(window=window, min_periods=min_periods).mean()
    std = s.rolling(window=window, min_periods=min_periods).std(ddof=0)
    z = (s - mean) / std.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)



def _pct_change_n(s: pd.Series, periods: int) -> pd.Series:
    return s.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)



def _normalize_score_components(weighted_sum: pd.Series, max_abs_weight: float) -> pd.Series:
    """Map weighted sum in [-max_abs_weight, +max_abs_weight] to [0, 100]."""
    scaled = 50.0 + 50.0 * (weighted_sum / max_abs_weight)
    return _clip100(scaled)


# -----------------------------------------------------------------------------
# Feature derivation helpers from price data
# -----------------------------------------------------------------------------


def _ensure_price_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = _series(out, "close")
    volume = _series(out, "volume")

    if "MA20" not in out.columns:
        out["MA20"] = close.rolling(20, min_periods=5).mean()
    if "MA50" not in out.columns:
        out["MA50"] = close.rolling(50, min_periods=10).mean()
    if "MA200" not in out.columns:
        out["MA200"] = close.rolling(200, min_periods=20).mean()

    if "RSI" not in out.columns:
        delta = close.diff()
        up = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        down = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs = up / down.replace(0, np.nan)
        out["RSI"] = 100 - (100 / (1 + rs))

    if "MACD" not in out.columns or "MACD_Signal" not in out.columns:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        out["MACD"] = out.get("MACD", macd)
        out["MACD_Signal"] = out.get("MACD_Signal", signal)

    if "BollingerUpper" not in out.columns or "BollingerLower" not in out.columns:
        ma20 = _series(out, "MA20")
        std20 = close.rolling(20, min_periods=5).std(ddof=0)
        out["BollingerUpper"] = out.get("BollingerUpper", ma20 + 2.0 * std20)
        out["BollingerLower"] = out.get("BollingerLower", ma20 - 2.0 * std20)

    if "OBV" not in out.columns:
        sign = np.sign(close.diff().fillna(0.0))
        out["OBV"] = (sign * volume.fillna(0.0)).cumsum()

    if "Volatility_30" not in out.columns:
        ret = close.pct_change()
        out["Volatility_30"] = ret.rolling(30, min_periods=10).std(ddof=0) * np.sqrt(252)

    if "abnormal_volume_z" not in out.columns:
        out["abnormal_volume_z"] = _rolling_zscore(np.log1p(volume.fillna(0.0)), window=60)

    if "ad_line" not in out.columns:
        high = _series(out, "high", default=np.nan)
        low = _series(out, "low", default=np.nan)
        money_flow_mult = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        out["ad_line"] = (money_flow_mult.fillna(0.0) * volume.fillna(0.0)).cumsum()

    return out


# -----------------------------------------------------------------------------
# Bucket scoring
# -----------------------------------------------------------------------------


def compute_insiders_score(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    net_buy = _series(df, "ins_net_buy_usd_30d", 0.0)
    buy_count = _series(df, "ins_buy_count_30d", 0.0)
    sell_count = _series(df, "ins_sell_count_30d", 0.0)
    buy_sell_ratio = _series(df, "ins_buy_sell_ratio_30d", 0.0)
    cluster_buy = _bool_series(df, "ins_cluster_buy_flag", 0.0)
    cluster_sell = _bool_series(df, "ins_cluster_sell_flag", 0.0)
    ceo_cfo_buy = _bool_series(df, "ins_ceo_cfo_buy_flag", 0.0)
    director_buy = _bool_series(df, "ins_director_buy_flag", 0.0)
    high_conviction = _bool_series(df, "ins_high_conviction_buy_flag", 0.0)
    ownership_delta = _series(df, "ins_ownership_delta_pct_30d", 0.0)
    tenb5_sell = _series(df, "ins_10b5_1_sell_usd_30d", 0.0)
    non_info = _series(df, "ins_noninformational_usd_30d", 0.0)
    first_buy_years = _bool_series(df, "ins_first_buy_in_years_flag", 0.0)

    weighted = (
        1.80 * _bounded_tanh(net_buy, 2_000_000.0)
        + 0.55 * _bounded_tanh(buy_count - sell_count, 3.0)
        + 0.60 * _bounded_tanh(buy_sell_ratio - 1.0, 1.0)
        + 0.90 * cluster_buy
        - 0.90 * cluster_sell
        + 0.90 * ceo_cfo_buy
        + 0.45 * director_buy
        + 1.00 * high_conviction
        + 0.75 * _bounded_tanh(ownership_delta, 1.0)
        - 0.55 * _bounded_tanh(tenb5_sell, 2_000_000.0)
        - 0.65 * _bounded_tanh(non_info, 1_500_000.0)
        + 0.55 * first_buy_years
    )
    score = _normalize_score_components(weighted, max_abs_weight=9.5)

    coverage_cols = [
        "ins_net_buy_usd_30d", "ins_buy_count_30d", "ins_sell_count_30d",
        "ins_buy_sell_ratio_30d", "ins_ownership_delta_pct_30d",
    ]
    conf = pd.Series(35.0, index=df.index, dtype=float)
    conf += df.reindex(columns=coverage_cols).notna().mean(axis=1).fillna(0.0) * 35.0
    conf += 15.0 * ((cluster_buy > 0) | (cluster_sell > 0)).astype(float)
    conf += 10.0 * high_conviction
    conf += 5.0 * ((ceo_cfo_buy > 0) | (director_buy > 0)).astype(float)
    conf = _clip100(conf)
    return score, conf



def compute_technicals_score(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    px = _ensure_price_indicators(df)
    close = _series(px, "close")
    ma20 = _series(px, "MA20")
    ma50 = _series(px, "MA50")
    ma200 = _series(px, "MA200")
    rsi = _series(px, "RSI")
    macd = _series(px, "MACD")
    macd_signal = _series(px, "MACD_Signal")
    bb_u = _series(px, "BollingerUpper")
    bb_l = _series(px, "BollingerLower")
    abn_vol = _series(px, "abnormal_volume_z")
    obv = _series(px, "OBV")
    ad_line = _series(px, "ad_line")

    trend20 = (close > ma20).astype(float)
    trend50 = (close > ma50).astype(float)
    trend200 = (close > ma200).astype(float)
    stack_bull = ((ma20 > ma50) & (ma50 > ma200)).astype(float)
    rsi_good = ((rsi >= 45) & (rsi <= 70)).astype(float)
    rsi_overbought = (rsi > 75).astype(float)
    macd_bull = (macd > macd_signal).astype(float)
    bb_position = ((close - bb_l) / (bb_u - bb_l).replace(0, np.nan)).clip(0, 1).fillna(0.5)
    bb_bull = ((bb_position >= 0.40) & (bb_position <= 0.85)).astype(float)
    volume_confirm = (abn_vol > 0.5).astype(float)
    obv_trend = (_pct_change_n(obv, 10) > 0).astype(float)
    ad_trend = (_pct_change_n(ad_line, 10) > 0).astype(float)

    weighted = (
        0.95 * trend20 + 1.00 * trend50 + 1.10 * trend200 + 1.20 * stack_bull
        + 0.85 * rsi_good - 0.80 * rsi_overbought
        + 1.05 * macd_bull + 0.80 * bb_bull
        + 0.70 * volume_confirm + 0.70 * obv_trend + 0.70 * ad_trend
    ) - 4.0
    score = _normalize_score_components(weighted, max_abs_weight=5.0)

    coverage_cols = ["close", "MA20", "MA50", "MA200", "RSI", "MACD", "MACD_Signal", "BollingerUpper", "BollingerLower"]
    conf = pd.Series(30.0, index=df.index, dtype=float)
    conf += px.reindex(columns=coverage_cols).notna().mean(axis=1).fillna(0.0) * 50.0
    conf += 10.0 * volume_confirm
    conf += 10.0 * ((stack_bull > 0) | (macd_bull > 0)).astype(float)
    conf = _clip100(conf)
    return score, conf



def compute_fundamentals_score(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    pe_rel = _series(df, "pe_relative_to_sector", np.nan)
    debt_ebitda = _series(df, "debt_to_ebitda", np.nan)
    roe = _series(df, "roe", np.nan)
    gross_margin_trend = _series(df, "gross_margin_trend", np.nan)
    eps_revision = _series(df, "eps_revision_score", np.nan)
    rev_revision = _series(df, "revenue_revision_score", np.nan)
    cash_runway_qtrs = _series(df, "cash_runway_qtrs", np.nan)
    dilution_risk = _series(df, "dilution_risk_score", np.nan)

    pe_good = np.where(np.isnan(pe_rel), 0.0, np.where(pe_rel <= 1.20, 1.0, np.where(pe_rel <= 1.50, 0.3, -0.7)))
    debt_good = np.where(np.isnan(debt_ebitda), 0.0, np.where(debt_ebitda < 4.0, 1.0, np.where(debt_ebitda < 5.5, 0.2, -0.9)))
    roe_good = np.where(np.isnan(roe), 0.0, np.where(roe > 10.0, 1.0, np.where(roe > 5.0, 0.2, -0.7)))
    margin_good = np.nan_to_num(np.tanh(gross_margin_trend / 3.0), nan=0.0)
    eps_good = np.nan_to_num(np.tanh(eps_revision / 2.0), nan=0.0)
    rev_good = np.nan_to_num(np.tanh(rev_revision / 2.0), nan=0.0)
    runway_good = np.where(np.isnan(cash_runway_qtrs), 0.0, np.where(cash_runway_qtrs >= 6.0, 0.8, np.where(cash_runway_qtrs >= 3.0, 0.2, -0.8)))
    dilution_bad = np.nan_to_num(np.tanh(dilution_risk / 2.0), nan=0.0)

    weighted = (
        1.10 * pe_good + 1.05 * debt_good + 1.05 * roe_good
        + 0.70 * margin_good + 0.75 * eps_good + 0.55 * rev_good
        + 0.55 * runway_good - 0.80 * dilution_bad
    )
    score = _normalize_score_components(pd.Series(weighted, index=df.index), max_abs_weight=6.0)

    coverage_cols = [
        "pe_relative_to_sector", "debt_to_ebitda", "roe", "gross_margin_trend",
        "eps_revision_score", "revenue_revision_score", "cash_runway_qtrs", "dilution_risk_score"
    ]
    conf = pd.Series(25.0, index=df.index, dtype=float)
    conf += df.reindex(columns=coverage_cols).notna().mean(axis=1).fillna(0.0) * 60.0
    conf += 5.0 * (~pd.isna(roe)).astype(float)
    conf += 10.0 * (~pd.isna(debt_ebitda)).astype(float)
    conf = _clip100(conf)
    return score, conf



def compute_flows_score(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    etf_flow_z = _series(df, "etf_flow_5d_z", 0.0)
    put_call = _series(df, "put_call_ratio", np.nan)
    iv_rank = _series(df, "iv_rank", np.nan)
    unusual_call = _bool_series(df, "unusual_call_flag", 0.0)
    unusual_put = _bool_series(df, "unusual_put_flag", 0.0)
    short_float = _series(df, "short_float_pct", np.nan)
    short_delta = _series(df, "short_float_delta", 0.0)
    inst_accum = _series(df, "inst_accumulation_score", 0.0)
    thirteenf = _series(df, "thirteenf_accumulation_score", 0.0)
    cta_proxy = _series(df, "cta_proxy_signal", 0.0)
    block_trade = _series(df, "block_trade_proxy", 0.0)

    put_call_good = np.where(np.isnan(put_call), 0.0, np.where(put_call < 0.85, 0.8, np.where(put_call < 1.10, 0.2, -0.8)))
    iv_good = np.where(np.isnan(iv_rank), 0.0, np.where(iv_rank < 80.0, 0.3, -0.4))
    short_penalty = np.where(np.isnan(short_float), 0.0, np.where(short_float < 15.0, 0.5, np.where(short_float < 25.0, -0.2, -0.8)))

    weighted = (
        0.95 * _bounded_tanh(etf_flow_z, 1.5)
        - 0.85 * pd.Series(put_call_good, index=df.index)
        - 0.25 * pd.Series(iv_good, index=df.index)
        + 0.95 * unusual_call - 0.95 * unusual_put
        + 0.45 * pd.Series(short_penalty, index=df.index)
        - 0.85 * _bounded_tanh(short_delta, 5.0)
        + 0.70 * _bounded_tanh(inst_accum, 2.0)
        + 0.45 * _bounded_tanh(thirteenf, 2.0)
        + 0.65 * _bounded_tanh(cta_proxy, 1.0)
        + 0.35 * _bounded_tanh(block_trade, 2.0)
    )
    score = _normalize_score_components(weighted, max_abs_weight=5.5)

    coverage_cols = [
        "etf_flow_5d_z", "put_call_ratio", "iv_rank", "short_float_pct",
        "short_float_delta", "inst_accumulation_score", "thirteenf_accumulation_score"
    ]
    conf = pd.Series(25.0, index=df.index, dtype=float)
    conf += df.reindex(columns=coverage_cols).notna().mean(axis=1).fillna(0.0) * 55.0
    conf += 10.0 * ((unusual_call > 0) | (unusual_put > 0)).astype(float)
    conf += 10.0 * (~pd.isna(short_float)).astype(float)
    conf = _clip100(conf)
    return score, conf



def compute_macro_score(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    dxy_z = _series(df, "dxy_z", 0.0)
    vix_z = _series(df, "vix_z", 0.0)
    ust2y = _series(df, "ust2y_change_5d", 0.0)
    ust10y = _series(df, "ust10y_change_5d", 0.0)
    real10y = _series(df, "real10y_change_5d", 0.0)
    cred = _series(df, "credit_spread_ig_change_5d", 0.0)
    curve = _series(df, "yield_curve_2s10s", 0.0)
    growth = _series(df, "macro_growth_regime", 0.0)
    inflation = _series(df, "macro_inflation_regime", 0.0)
    risk_on = _bool_series(df, "risk_on_flag", 0.0)

    weighted = (
        -0.90 * _bounded_tanh(dxy_z, 1.5)
        -1.00 * _bounded_tanh(vix_z, 1.5)
        -0.45 * _bounded_tanh(ust2y, 0.35)
        -0.55 * _bounded_tanh(ust10y, 0.35)
        -0.95 * _bounded_tanh(real10y, 0.25)
        -0.90 * _bounded_tanh(cred, 0.30)
        +0.55 * _bounded_tanh(curve, 0.75)
        +0.75 * _bounded_tanh(growth, 1.0)
        -0.35 * _bounded_tanh(inflation, 1.0)
        +0.95 * risk_on
    )
    score = _normalize_score_components(weighted, max_abs_weight=6.0)

    coverage_cols = [
        "dxy_z", "vix_z", "ust2y_change_5d", "ust10y_change_5d",
        "real10y_change_5d", "credit_spread_ig_change_5d",
        "yield_curve_2s10s", "macro_growth_regime", "macro_inflation_regime",
        "risk_on_flag"
    ]
    conf = pd.Series(30.0, index=df.index, dtype=float)
    conf += df.reindex(columns=coverage_cols).notna().mean(axis=1).fillna(0.0) * 60.0
    conf += 10.0 * (~pd.isna(vix_z)).astype(float)
    conf = _clip100(conf)
    return score, conf


# -----------------------------------------------------------------------------
# Master scoring function
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MosaicWeights:
    insiders: float = 0.35
    technicals: float = 0.20
    fundamentals: float = 0.20
    flows: float = 0.15
    macro: float = 0.10


DEFAULT_WEIGHTS = MosaicWeights()



def compute_mosaic_scores(
    df: pd.DataFrame,
    weights: MosaicWeights = DEFAULT_WEIGHTS,
    bullish_threshold_raw: float = 6.0,
) -> pd.DataFrame:
    """Compute Mosaic Theory scores from a point-in-time feature frame.

    Parameters
    ----------
    df:
        Input DataFrame. Best practice is one row per ticker-date.
    weights:
        Bucket weights, defaulting to insider-weighted framework.
    bullish_threshold_raw:
        Threshold on the -10..+10 raw score for a bullish screen pass.

    Returns
    -------
    pd.DataFrame
        Original frame plus bucket scores, confidence, rating and helper flags.
    """
    out = df.copy()
    if "date" in out.columns:
        out["date"] = _to_datetime_series(out["date"])

    ins_score, ins_conf = compute_insiders_score(out)
    tech_score, tech_conf = compute_technicals_score(out)
    fund_score, fund_conf = compute_fundamentals_score(out)
    flow_score, flow_conf = compute_flows_score(out)
    macro_score, macro_conf = compute_macro_score(out)

    out["mosaic_insiders_score"] = ins_score.values
    out["mosaic_technicals_score"] = tech_score.values
    out["mosaic_fundamentals_score"] = fund_score.values
    out["mosaic_flows_score"] = flow_score.values
    out["mosaic_macro_score"] = macro_score.values

    out["mosaic_insiders_confidence"] = ins_conf.values
    out["mosaic_technicals_confidence"] = tech_conf.values
    out["mosaic_fundamentals_confidence"] = fund_conf.values
    out["mosaic_flows_confidence"] = flow_conf.values
    out["mosaic_macro_confidence"] = macro_conf.values

    out["mosaic_total_score"] = (
        weights.insiders * out["mosaic_insiders_score"]
        + weights.technicals * out["mosaic_technicals_score"]
        + weights.fundamentals * out["mosaic_fundamentals_score"]
        + weights.flows * out["mosaic_flows_score"]
        + weights.macro * out["mosaic_macro_score"]
    )

    out["mosaic_score_raw"] = ((out["mosaic_total_score"] - 50.0) / 50.0) * 10.0

    hv30 = _series(out, "Volatility_30", np.nan)
    if hv30.isna().all() and "close" in out.columns:
        close = _series(out, "close")
        hv30 = close.pct_change().rolling(30, min_periods=10).std(ddof=0) * np.sqrt(252)
        out["Volatility_30"] = hv30
    hv30 = hv30.replace(0, np.nan)
    out["mosaic_vol_adjusted_score"] = out["mosaic_score_raw"] * (1.0 / hv30.clip(lower=0.05))

    bucket_conf_cols = [
        "mosaic_insiders_confidence",
        "mosaic_technicals_confidence",
        "mosaic_fundamentals_confidence",
        "mosaic_flows_confidence",
        "mosaic_macro_confidence",
    ]
    bucket_score_cols = [
        "mosaic_insiders_score",
        "mosaic_technicals_score",
        "mosaic_fundamentals_score",
        "mosaic_flows_score",
        "mosaic_macro_score",
    ]

    bucket_dispersion = out[bucket_score_cols].std(axis=1).fillna(0.0)
    score_agreement_bonus = np.clip(20.0 - bucket_dispersion, 0.0, 20.0)
    coverage_bonus = out[bucket_conf_cols].mean(axis=1).fillna(0.0) * 0.20
    out["mosaic_confidence"] = _clip100(coverage_bonus + score_agreement_bonus).values

    out["mosaic_supporting_tiles_count"] = (
        out[bucket_score_cols].ge(60.0).sum(axis=1) + out[bucket_score_cols].le(40.0).sum(axis=1)
    ).astype(int)

    out["mosaic_has_catalyst_90d"] = (
        (_series(out, "days_to_earnings", np.nan).between(0, 90, inclusive="both"))
        | (_series(out, "days_to_fomc", np.nan).between(0, 90, inclusive="both"))
        | (_series(out, "days_to_sector_event", np.nan).between(0, 90, inclusive="both"))
        | (_bool_series(out, "has_major_catalyst_30d", 0.0) > 0)
    ).astype(int)

    out["mosaic_pass_bullish_threshold"] = (out["mosaic_score_raw"] > bullish_threshold_raw).astype(int)

    conditions = [
        out["mosaic_score_raw"] > 5.0,
        out["mosaic_score_raw"].between(-5.0, 5.0, inclusive="both"),
        out["mosaic_score_raw"] < -5.0,
    ]
    choices = ["Bullish", "Neutral", "Bearish"]
    out["mosaic_rating"] = np.select(conditions, choices, default="Neutral")

    return out


# -----------------------------------------------------------------------------
# Merge helper
# -----------------------------------------------------------------------------


MOSAIC_OUTPUT_COLUMNS: List[str] = [
    "mosaic_insiders_score",
    "mosaic_technicals_score",
    "mosaic_fundamentals_score",
    "mosaic_flows_score",
    "mosaic_macro_score",
    "mosaic_insiders_confidence",
    "mosaic_technicals_confidence",
    "mosaic_fundamentals_confidence",
    "mosaic_flows_confidence",
    "mosaic_macro_confidence",
    "mosaic_total_score",
    "mosaic_score_raw",
    "mosaic_vol_adjusted_score",
    "mosaic_confidence",
    "mosaic_supporting_tiles_count",
    "mosaic_has_catalyst_90d",
    "mosaic_pass_bullish_threshold",
    "mosaic_rating",
]



def merge_mosaic_features(price_df: pd.DataFrame, mosaic_df: pd.DataFrame) -> pd.DataFrame:
    left = price_df.copy()
    right = mosaic_df.copy()
    if "date" not in left.columns or "date" not in right.columns:
        raise ValueError("Both DataFrames must contain a 'date' column for merge.")
    left["date"] = _to_datetime_series(left["date"]).dt.normalize()
    right["date"] = _to_datetime_series(right["date"]).dt.normalize()

    keep_cols = [c for c in ["date", *MOSAIC_OUTPUT_COLUMNS] if c in right.columns]
    merged = left.merge(right[keep_cols], on="date", how="left")
    for col in MOSAIC_OUTPUT_COLUMNS:
        if col in merged.columns and col != "mosaic_rating":
            merged[col] = merged[col].ffill()
    if "mosaic_rating" in merged.columns:
        merged["mosaic_rating"] = merged["mosaic_rating"].ffill().fillna("Neutral")
    return merged


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------



def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute daily Mosaic Theory scores from a feature CSV.")
    p.add_argument("--input-csv", required=True, help="Input CSV with daily ticker features.")
    p.add_argument("--output-csv", required=True, help="Output CSV path for scored features.")
    p.add_argument(
        "--bullish-threshold-raw",
        type=float,
        default=6.0,
        help="Bullish threshold on the -10..+10 raw Mosaic score. Default: 6.0",
    )
    return p



def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    scored = compute_mosaic_scores(df, bullish_threshold_raw=args.bullish_threshold_raw)
    scored.to_csv(args.output_csv, index=False)
    print(f"Saved scored Mosaic features to {args.output_csv}")


if __name__ == "__main__":
    main()
