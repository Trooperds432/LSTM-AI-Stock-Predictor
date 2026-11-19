"""
macro_cross_asset.py
====================

Point-in-time, lag-aware macro / cross-asset feature engineering for equity ML
pipelines.

Purpose
-------
This module implements the *macro / cross-asset* block of an institutional
Mosaic Theory pipeline. It pulls public market and macro series, aligns them to
tradable dates, and produces daily model features with a clean merge path into a
price / indicator frame.

Public-source-only inputs used here
-----------------------------------
- Yahoo Finance chart endpoints (public) for:
  - DXY proxy: DX-Y.NYB
  - VIX: ^VIX
- FRED public CSV endpoints for:
  - DGS2    : 2Y Treasury constant maturity yield
  - DGS10   : 10Y Treasury constant maturity yield
  - DFII10  : 10Y Treasury inflation-indexed security, constant maturity
  - BAMLC0A0CM : ICE BofA US Corporate Index OAS (investment grade spread proxy)
  - CPIAUCSL   : CPI level (used with an explicit publication-lag assumption)
  - INDPRO     : Industrial production index (used with an explicit publication-
                 lag assumption)

Design choices
--------------
- Public-source only.
- No API key required.
- Daily market series are aligned by date.
- Monthly macro series used for regime detection are shifted by configurable
  *availability lags* so they are not leaked into the past.
- Output includes the exact feature columns requested by the user plus a few
  diagnostics for debugging and model analysis.

Exact output columns
--------------------
- dxy_z
- vix_z
- ust2y_change_5d
- ust10y_change_5d
- real10y_change_5d
- credit_spread_ig_change_5d
- yield_curve_2s10s
- macro_growth_regime
- macro_inflation_regime
- risk_on_flag

Notes on regime features
------------------------
`macro_growth_regime` and `macro_inflation_regime` are encoded as:
- +1 : accelerating
-  0 : neutral / indeterminate
- -1 : decelerating

`risk_on_flag` is encoded as:
- 1 : risk-on
- 0 : otherwise

This module intentionally focuses on the *core macro cross-asset block*.
Sector-specific FX, futures curves, options/flows, and event calendars should
live in separate feature builders.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
DEFAULT_TIMEOUT = 30

DXY_YAHOO_SYMBOL = "DX-Y.NYB"
VIX_YAHOO_SYMBOL = "^VIX"

FRED_SERIES = {
    "ust2y": "DGS2",
    "ust10y": "DGS10",
    "real10y": "DFII10",
    "credit_spread_ig": "BAMLC0A0CM",
    "cpi": "CPIAUCSL",
    "indpro": "INDPRO",
}

EXACT_FEATURE_COLUMNS: List[str] = [
    "dxy_z",
    "vix_z",
    "ust2y_change_5d",
    "ust10y_change_5d",
    "real10y_change_5d",
    "credit_spread_ig_change_5d",
    "yield_curve_2s10s",
    "macro_growth_regime",
    "macro_inflation_regime",
    "risk_on_flag",
]

DIAGNOSTIC_COLUMNS: List[str] = [
    "dxy_level",
    "vix_level",
    "ust2y_level",
    "ust10y_level",
    "real10y_level",
    "credit_spread_ig_level",
    "growth_yoy_proxy",
    "inflation_yoy_proxy",
    "growth_delta_3m",
    "inflation_delta_3m",
    "macro_data_freshness_days",
    "macro_source_row_count",
]

ALL_OUTPUT_COLUMNS: List[str] = EXACT_FEATURE_COLUMNS + DIAGNOSTIC_COLUMNS


@dataclass(frozen=True)
class MacroConfig:
    """Configuration for macro / cross-asset feature generation."""

    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    zscore_lookback: int = 252
    change_lookback_days: int = 5
    min_zscore_periods: int = 60
    cpi_publication_lag_days: int = 14
    indpro_publication_lag_days: int = 16
    regime_neutral_band: float = 0.25
    timeout_seconds: int = DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        if self.zscore_lookback <= 1:
            raise ValueError("zscore_lookback must be > 1")
        if self.change_lookback_days <= 0:
            raise ValueError("change_lookback_days must be positive")
        if self.min_zscore_periods <= 1:
            raise ValueError("min_zscore_periods must be > 1")


def _to_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def make_session() -> requests.Session:
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 macro-cross-asset-feature-builder",
            "Accept": "application/json,text/csv,text/plain,*/*",
        }
    )
    return session


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        if pd.isna(value):
            return None
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {".", "nan", "NaN", "None"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_daily_frame(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out = out.dropna(subset=[date_col])
    out = out.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
    return out.reset_index(drop=True)


def fetch_yahoo_daily_series(
    session: requests.Session,
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    timeout_seconds: int = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """Fetch a public Yahoo Finance chart series without requiring yfinance."""
    period1 = int((start_date - pd.Timedelta(days=7)).timestamp())
    period2 = int((end_date + pd.Timedelta(days=7)).timestamp())
    url = YAHOO_CHART_URL.format(symbol=symbol)
    resp = session.get(
        url,
        params={
            "interval": "1d",
            "period1": period1,
            "period2": period2,
            "includePrePost": "false",
            "events": "div,splits,capitalGains",
        },
        timeout=timeout_seconds,
    )
    resp.raise_for_status()
    payload = resp.json()

    result = payload.get("chart", {}).get("result", [])
    if not result:
        error = payload.get("chart", {}).get("error")
        raise RuntimeError(f"Yahoo Finance returned no result for {symbol}: {error}")

    item = result[0]
    ts_list = item.get("timestamp") or []
    quote = ((item.get("indicators") or {}).get("quote") or [{}])[0]
    adjclose_block = ((item.get("indicators") or {}).get("adjclose") or [{}])[0]

    close = quote.get("close") or []
    adjclose = adjclose_block.get("adjclose") or close

    rows: List[Dict[str, Any]] = []
    for ts_value, px in zip(ts_list, adjclose):
        val = _safe_float(px)
        if val is None:
            continue
        rows.append(
            {
                "date": pd.to_datetime(ts_value, unit="s", utc=True).tz_convert(None).normalize(),
                "value": float(val),
                "source_symbol": symbol,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Yahoo Finance returned an empty price series for {symbol}")
    df = _normalize_daily_frame(df)
    return df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)


def fetch_fred_series(
    session: requests.Session,
    series_id: str,
    timeout_seconds: int = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """Fetch a public FRED CSV series."""
    url = FRED_CSV_URL.format(series_id=series_id)
    resp = session.get(url, timeout=timeout_seconds)
    resp.raise_for_status()
    df = pd.read_csv(pd.io.common.StringIO(resp.text))
    if df.empty:
        raise RuntimeError(f"FRED returned an empty series for {series_id}")
    df.columns = [c.strip() for c in df.columns]
    if series_id not in df.columns:
        raise RuntimeError(f"FRED CSV for {series_id} did not contain expected column")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["DATE"], errors="coerce").dt.normalize(),
            "value": pd.to_numeric(df[series_id], errors="coerce"),
            "source_symbol": series_id,
        }
    ).dropna(subset=["date"])
    out = _normalize_daily_frame(out)
    return out.reset_index(drop=True)


def _date_window_from_price_frame(
    price_df: Optional[pd.DataFrame],
    config: MacroConfig,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if config.start_date is not None and config.end_date is not None:
        return _to_ts(config.start_date), _to_ts(config.end_date)

    if price_df is None or price_df.empty or "date" not in price_df.columns:
        end_date = _to_ts(pd.Timestamp.utcnow()) if config.end_date is None else _to_ts(config.end_date)
        start_date = end_date - pd.Timedelta(days=730) if config.start_date is None else _to_ts(config.start_date)
        return start_date, end_date

    dates = pd.to_datetime(price_df["date"], errors="coerce").dropna().dt.normalize()
    if dates.empty:
        raise ValueError("price_df has no valid dates")

    start_date = _to_ts(config.start_date) if config.start_date is not None else dates.min()
    end_date = _to_ts(config.end_date) if config.end_date is not None else dates.max()
    return start_date, end_date


def _add_publication_lag(monthly_df: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """
    Convert an observation-dated monthly series into an availability-dated series.

    This is a best-effort lag-safety transform used for slow macro proxies like CPI
    and Industrial Production. The series value is assumed tradable only after the
    configured publication lag from the observation month-end.
    """
    out = monthly_df.copy()
    out = _normalize_daily_frame(out)
    obs_date = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    # Public availability date approximated as month-end + lag.
    availability_date = obs_date + pd.offsets.MonthEnd(0) + pd.to_timedelta(lag_days, unit="D")
    out["date"] = pd.to_datetime(availability_date).dt.normalize()
    out = _normalize_daily_frame(out)
    return out


def _rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    z = (series - mean) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def _classify_regime(delta: pd.Series, neutral_band: float) -> pd.Series:
    out = pd.Series(0, index=delta.index, dtype=float)
    out = out.mask(delta > neutral_band, 1.0)
    out = out.mask(delta < -neutral_band, -1.0)
    return out.astype(int)


def _business_day_calendar(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    cal = pd.date_range(start=start_date, end=end_date, freq="B")
    return pd.DataFrame({"date": cal.normalize()})


def build_macro_feature_frame(
    price_df: Optional[pd.DataFrame] = None,
    config: Optional[MacroConfig] = None,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Build daily macro / cross-asset features.

    Parameters
    ----------
    price_df:
        Optional market-date frame with at least a `date` column. If provided, the
        output is aligned to those dates. Otherwise a business-day calendar is used.
    config:
        MacroConfig with lookbacks, lags, and date bounds.
    session:
        Optional requests session for dependency injection / testing.
    """
    config = config or MacroConfig()
    session = session or make_session()

    start_date, end_date = _date_window_from_price_frame(price_df, config)
    fetch_start = start_date - pd.Timedelta(days=max(450, config.zscore_lookback + 120))
    fetch_end = end_date + pd.Timedelta(days=7)

    LOGGER.info("Fetching macro market series from %s to %s", fetch_start.date(), fetch_end.date())

    dxy = fetch_yahoo_daily_series(session, DXY_YAHOO_SYMBOL, fetch_start, fetch_end, config.timeout_seconds)
    vix = fetch_yahoo_daily_series(session, VIX_YAHOO_SYMBOL, fetch_start, fetch_end, config.timeout_seconds)

    fred_frames: Dict[str, pd.DataFrame] = {}
    for key, series_id in FRED_SERIES.items():
        fred_frames[key] = fetch_fred_series(session, series_id, config.timeout_seconds)

    # Lag-safe monthly series for regime proxies. Compute YoY on the native monthly
    # observation series first, then shift the resulting signal to its assumed public
    # availability date.
    cpi_monthly = fred_frames["cpi"].copy()
    cpi_monthly["value"] = (cpi_monthly["value"] / cpi_monthly["value"].shift(12) - 1.0) * 100.0
    cpi_pub = _add_publication_lag(cpi_monthly, config.cpi_publication_lag_days)

    indpro_monthly = fred_frames["indpro"].copy()
    indpro_monthly["value"] = (indpro_monthly["value"] / indpro_monthly["value"].shift(12) - 1.0) * 100.0
    indpro_pub = _add_publication_lag(indpro_monthly, config.indpro_publication_lag_days)

    calendar = (
        _normalize_daily_frame(price_df[["date"]]) if price_df is not None and not price_df.empty else _business_day_calendar(start_date, end_date)
    )

    macro = calendar.copy()
    series_map = {
        "dxy_level": dxy,
        "vix_level": vix,
        "ust2y_level": fred_frames["ust2y"],
        "ust10y_level": fred_frames["ust10y"],
        "real10y_level": fred_frames["real10y"],
        "credit_spread_ig_level": fred_frames["credit_spread_ig"],
        "cpi_available": cpi_pub,
        "indpro_available": indpro_pub,
    }

    for col_name, df in series_map.items():
        tmp = df[["date", "value"]].rename(columns={"value": col_name})
        macro = macro.merge(tmp, on="date", how="left")

    # Forward-fill slower and daily macro series onto the market-date calendar.
    macro = macro.sort_values("date").reset_index(drop=True)
    fill_cols = [c for c in macro.columns if c != "date"]
    macro[fill_cols] = macro[fill_cols].ffill()

    # Core features.
    macro["dxy_z"] = _rolling_zscore(macro["dxy_level"], config.zscore_lookback, config.min_zscore_periods)
    macro["vix_z"] = _rolling_zscore(macro["vix_level"], config.zscore_lookback, config.min_zscore_periods)

    macro["ust2y_change_5d"] = macro["ust2y_level"].diff(config.change_lookback_days)
    macro["ust10y_change_5d"] = macro["ust10y_level"].diff(config.change_lookback_days)
    macro["real10y_change_5d"] = macro["real10y_level"].diff(config.change_lookback_days)
    macro["credit_spread_ig_change_5d"] = macro["credit_spread_ig_level"].diff(config.change_lookback_days)
    macro["yield_curve_2s10s"] = macro["ust10y_level"] - macro["ust2y_level"]

    # Growth / inflation regime proxies.
    macro["growth_yoy_proxy"] = macro["indpro_available"]
    macro["inflation_yoy_proxy"] = macro["cpi_available"]

    # Monthly series are forward-filled daily after publication lag. A 63-business-day
    # delta approximates roughly one quarter of daily market time.
    macro["growth_delta_3m"] = macro["growth_yoy_proxy"] - macro["growth_yoy_proxy"].shift(63)
    macro["inflation_delta_3m"] = macro["inflation_yoy_proxy"] - macro["inflation_yoy_proxy"].shift(63)

    macro["macro_growth_regime"] = _classify_regime(macro["growth_delta_3m"], config.regime_neutral_band)
    macro["macro_inflation_regime"] = _classify_regime(macro["inflation_delta_3m"], config.regime_neutral_band)

    # Risk-on heuristic: lower-than-normal vol, tighter credit, and no sharp USD squeeze.
    # This is an intentionally simple public-data proxy for use as a model feature.
    risk_on_conditions = (
        (macro["vix_z"] < 0.25)
        & (macro["credit_spread_ig_change_5d"] <= 0.0)
        & (macro["dxy_z"] < 1.0)
    )
    macro["risk_on_flag"] = risk_on_conditions.fillna(False).astype(int)

    # Diagnostics.
    last_dates = pd.concat(
        [
            dxy[["date"]].assign(series="dxy"),
            vix[["date"]].assign(series="vix"),
            fred_frames["ust2y"][["date"]].assign(series="ust2y"),
            fred_frames["ust10y"][["date"]].assign(series="ust10y"),
            fred_frames["real10y"][["date"]].assign(series="real10y"),
            fred_frames["credit_spread_ig"][["date"]].assign(series="credit_spread_ig"),
            cpi_pub[["date"]].assign(series="cpi_pub"),
            indpro_pub[["date"]].assign(series="indpro_pub"),
        ],
        ignore_index=True,
    )
    latest_source_date = pd.to_datetime(last_dates["date"], errors="coerce").max()
    macro["macro_data_freshness_days"] = (macro["date"] - latest_source_date).dt.days.abs()
    macro["macro_source_row_count"] = int(
        len(dxy)
        + len(vix)
        + sum(len(df) for df in fred_frames.values())
        + len(cpi_pub)
        + len(indpro_pub)
    )

    # Trim to requested window.
    macro = macro[(macro["date"] >= start_date) & (macro["date"] <= end_date)].reset_index(drop=True)

    # Ensure all output columns exist.
    for col in ALL_OUTPUT_COLUMNS:
        if col not in macro.columns:
            macro[col] = np.nan

    return macro[["date"] + ALL_OUTPUT_COLUMNS].copy()


def merge_macro_features_into_price_frame(
    price_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    *,
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """Merge generated macro features into a price / indicator frame on `date`."""
    if "date" not in price_df.columns:
        raise ValueError("price_df must contain a 'date' column")
    if "date" not in macro_df.columns:
        raise ValueError("macro_df must contain a 'date' column")

    left = _normalize_daily_frame(price_df)
    right = _normalize_daily_frame(macro_df)
    merged = left.merge(right, on="date", how="left")

    cols = [c for c in macro_df.columns if c != "date"]
    if fill_method == "ffill":
        merged[cols] = merged[cols].ffill()
    elif fill_method == "zero":
        merged[cols] = merged[cols].fillna(0.0)
    elif fill_method != "none":
        raise ValueError("fill_method must be one of: 'ffill', 'zero', 'none'")

    return merged


def build_and_merge_macro_features(
    price_df: pd.DataFrame,
    config: Optional[MacroConfig] = None,
) -> pd.DataFrame:
    """Convenience wrapper: build macro features and merge into a price frame."""
    macro_df = build_macro_feature_frame(price_df=price_df, config=config)
    return merge_macro_features_into_price_frame(price_df, macro_df)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build macro / cross-asset features for an equity ML pipeline.")
    parser.add_argument("--price-csv", type=str, default=None, help="Optional CSV with at least a 'date' column.")
    parser.add_argument("--output-features", type=str, default=None, help="Optional path to write the macro feature CSV.")
    parser.add_argument("--output-merged", type=str, default=None, help="Optional path to write the merged price+macro CSV.")
    parser.add_argument("--start-date", type=str, default=None, help="Optional start date if no price CSV is supplied.")
    parser.add_argument("--end-date", type=str, default=None, help="Optional end date if no price CSV is supplied.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    return parser.parse_args()


def _load_price_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Expected a 'date' column in {path}")
    return df


def _config_from_args(args: argparse.Namespace) -> MacroConfig:
    start_date = _to_ts(args.start_date) if args.start_date else None
    end_date = _to_ts(args.end_date) if args.end_date else None
    return MacroConfig(start_date=start_date, end_date=end_date)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    price_df = _load_price_csv(args.price_csv)
    config = _config_from_args(args)

    macro_df = build_macro_feature_frame(price_df=price_df, config=config)

    if args.output_features:
        macro_df.to_csv(args.output_features, index=False)
        LOGGER.info("Wrote macro feature frame to %s", args.output_features)

    if args.output_merged:
        if price_df is None:
            raise ValueError("--output-merged requires --price-csv")
        merged = merge_macro_features_into_price_frame(price_df, macro_df)
        merged.to_csv(args.output_merged, index=False)
        LOGGER.info("Wrote merged frame to %s", args.output_merged)

    if not args.output_features and not args.output_merged:
        preview = macro_df.tail(10)
        print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
