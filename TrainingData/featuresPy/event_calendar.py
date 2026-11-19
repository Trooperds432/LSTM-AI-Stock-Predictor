"""
event_calendar.py
=================

Point-in-time, lag-safe event-calendar feature engineering for equity ML
pipelines.

Purpose
-------
This module implements the *event calendar* block of an institutional Mosaic
Theory pipeline. It builds daily catalyst features aligned to tradable dates and
produces a clean merge path into a price / indicator frame.

Public-source-only inputs used here
-----------------------------------
Core sources implemented in-code:
- Yahoo Finance quoteSummary/calendarEvents endpoint (public) for earnings date
- Federal Reserve FOMC calendar page (public) for scheduled FOMC meeting dates

Optional user-supplied supplemental catalysts:
- CSV of sector / ticker / macro events from public sources such as FDA, EIA,
  USDA, OPEC calendars, investor relations pages, conference schedules, etc.

Why the supplemental path exists
--------------------------------
There is no single clean, key-free public API that reliably covers all sector-
specific catalysts (PDUFA, OPEC, WASDE, refinery maintenance windows, lockups,
conference schedules, debt maturities, product launches, and similar items).
To avoid inventing dates or relying on brittle assumptions, this module:

1. Fetches what is cleanly available and broadly reliable (earnings, FOMC), and
2. Lets you merge additional *publicly sourced* catalyst rows through a simple,
   documented CSV schema.

Design choices
--------------
- Public-source only.
- No API key required.
- Features are aligned by event knowledge / public schedule date.
- Uses event dates, not realized outcomes.
- Does not infer MNPI.
- Does not fabricate unavailable sector event dates.

Exact output columns
--------------------
- days_to_earnings
- days_to_fomc
- days_to_sector_event
- has_major_catalyst_30d
- post_earnings_5d_flag

Feature semantics
-----------------
- `days_to_earnings`:
    Number of calendar days until the next known earnings date.
- `days_to_fomc`:
    Number of calendar days until the next scheduled FOMC meeting start date.
- `days_to_sector_event`:
    Number of calendar days until the next applicable sector / ticker catalyst
    from the supplemental event calendar (or built-in parsers, when available).
- `has_major_catalyst_30d`:
    1 if earnings, FOMC, or a major supplemental sector event occurs within the
    next 30 calendar days, else 0.
- `post_earnings_5d_flag`:
    1 during the five calendar days immediately after a known earnings date,
    else 0.

Supplemental event CSV schema
-----------------------------
Columns understood by this module:
- date               : required, event date
- event_name         : optional, free-text label
- event_type         : optional, e.g. sector_event, macro, product_launch,
                       conference, pdufa, opec, wasde, cpi, lockup
- ticker             : optional, specific ticker symbol
- sector             : optional, e.g. energy, healthcare, technology, finance,
                       commodities
- is_major           : optional, 1/0 or true/false
- source             : optional, public-source provenance note / URL
- notes              : optional

Filtering logic for supplemental events
---------------------------------------
An event applies if either:
- `ticker` matches the requested ticker, or
- `sector` matches the requested sector, or
- both are blank and it is a generic market / macro catalyst.

Recommended usage
-----------------
1. Build / load your market-date DataFrame for the ticker.
2. Call `build_event_calendar_feature_frame(...)`.
3. Merge on `date` into your training dataset.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (compatible; EventCalendarFeatureBuilder/1.0; "
    "+https://openai.com)"
)

SECTOR_ALIASES = {
    "tech": "technology",
    "information technology": "technology",
    "it": "technology",
    "health care": "healthcare",
    "biotech": "healthcare",
    "pharma": "healthcare",
    "financials": "finance",
    "banks": "finance",
    "banking": "finance",
    "oil": "energy",
    "gas": "energy",
    "materials": "commodities",
    "metals": "commodities",
    "mining": "commodities",
    "ag": "commodities",
}

FOMC_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)


@dataclass(frozen=True)
class EventRecord:
    """Canonical event row used before feature construction."""

    date: pd.Timestamp
    event_name: str
    event_type: str
    is_major: bool
    ticker: Optional[str] = None
    sector: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None


# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------

def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------

def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-")


def normalize_sector(sector: Optional[str]) -> Optional[str]:
    if sector is None or pd.isna(sector):
        return None
    s = str(sector).strip().lower()
    return SECTOR_ALIASES.get(s, s)


def _to_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value is None or value == "":
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "major"}


# -----------------------------------------------------------------------------
# Public-source fetchers
# -----------------------------------------------------------------------------

def fetch_yahoo_earnings_date(
    ticker: str,
    session: Optional[requests.Session] = None,
) -> Optional[pd.Timestamp]:
    """
    Fetch the next known earnings date from Yahoo Finance calendarEvents.

    This is a *forward-looking public schedule* and therefore safe to use as an
    event-calendar feature. If unavailable, returns None.
    """
    own_session = session is None
    session = session or _build_session()
    try:
        url = (
            "https://query2.finance.yahoo.com/v10/finance/quoteSummary/"
            f"{normalize_ticker(ticker)}?modules=calendarEvents"
        )
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        payload = resp.json()

        result = payload.get("quoteSummary", {}).get("result") or []
        if not result:
            return None
        cal = result[0].get("calendarEvents", {})
        earnings = cal.get("earnings", {})
        earnings_dates = earnings.get("earningsDate") or []
        if not earnings_dates:
            return None

        # Yahoo often returns one or two candidate dates.
        timestamps: List[pd.Timestamp] = []
        for item in earnings_dates:
            fmt = item.get("fmt") if isinstance(item, dict) else None
            raw = item.get("raw") if isinstance(item, dict) else None
            ts = _to_timestamp(fmt)
            if ts is None and raw is not None:
                ts = pd.Timestamp(pd.to_datetime(raw, unit="s", errors="coerce")).normalize()
                if pd.isna(ts):
                    ts = None
            if ts is not None:
                timestamps.append(ts)

        if not timestamps:
            return None
        return min(timestamps)
    except Exception as exc:  # pragma: no cover - network-dependent
        LOGGER.warning("Failed to fetch Yahoo earnings date for %s: %s", ticker, exc)
        return None
    finally:
        if own_session:
            session.close()



def fetch_fomc_calendar(
    start_date: Any,
    end_date: Any,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Fetch scheduled FOMC meeting dates from the Federal Reserve calendar page.

    Notes
    -----
    - The page contains yearly meeting date strings such as "January 28-29".
    - We parse the *first day* of each meeting window as the catalyst date.
    - If the page layout changes, this function fails closed and returns an
      empty DataFrame rather than hallucinating dates.
    """
    start_ts = _to_timestamp(start_date) or pd.Timestamp.today().normalize()
    end_ts = _to_timestamp(end_date) or (start_ts + pd.Timedelta(days=365))

    own_session = session is None
    session = session or _build_session()
    rows: List[EventRecord] = []

    try:
        url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        html = resp.text

        # Split page into year sections so month/day strings inherit the correct year.
        year_blocks = re.finditer(
            r"(?P<year>20\d{2})\s+FOMC Meetings(?P<body>.*?)(?=(20\d{2})\s+FOMC Meetings|$)",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        for block in year_blocks:
            year = int(block.group("year"))
            body = block.group("body")
            pattern = re.compile(
                rf"(?P<month>{FOMC_MONTHS})\s+"
                r"(?P<day1>\d{1,2})(?:\s*[–\-]\s*(?P<day2>\d{1,2}))?",
                flags=re.IGNORECASE,
            )
            for m in pattern.finditer(body):
                month_name = m.group("month")
                day1 = int(m.group("day1"))
                try:
                    dt = pd.Timestamp(f"{month_name} {day1}, {year}").normalize()
                except Exception:
                    continue
                if start_ts <= dt <= end_ts:
                    rows.append(
                        EventRecord(
                            date=dt,
                            event_name="FOMC meeting",
                            event_type="fomc",
                            is_major=True,
                            ticker=None,
                            sector=None,
                            source=url,
                            notes="Parsed from Federal Reserve FOMC calendar page",
                        )
                    )

        # Deduplicate in case regex hits repeated text.
        df = pd.DataFrame([r.__dict__ for r in rows]).drop_duplicates(subset=["date", "event_type"])
        if df.empty:
            return pd.DataFrame(columns=["date", "event_name", "event_type", "is_major", "source", "notes"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception as exc:  # pragma: no cover - network-dependent
        LOGGER.warning("Failed to fetch FOMC calendar: %s", exc)
        return pd.DataFrame(columns=["date", "event_name", "event_type", "is_major", "source", "notes"])
    finally:
        if own_session:
            session.close()


# -----------------------------------------------------------------------------
# Supplemental event ingestion
# -----------------------------------------------------------------------------

def load_supplemental_events_csv(path: Optional[str]) -> pd.DataFrame:
    """Load user-supplied catalyst rows from CSV."""
    if not path:
        return pd.DataFrame(
            columns=["date", "event_name", "event_type", "ticker", "sector", "is_major", "source", "notes"]
        )

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(
            columns=["date", "event_name", "event_type", "ticker", "sector", "is_major", "source", "notes"]
        )

    if "date" not in df.columns:
        raise ValueError("Supplemental event CSV must contain a 'date' column.")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out[out["date"].notna()].copy()

    if "event_name" not in out.columns:
        out["event_name"] = "supplemental_event"
    if "event_type" not in out.columns:
        out["event_type"] = "sector_event"
    if "ticker" not in out.columns:
        out["ticker"] = None
    if "sector" not in out.columns:
        out["sector"] = None
    if "is_major" not in out.columns:
        out["is_major"] = False
    if "source" not in out.columns:
        out["source"] = None
    if "notes" not in out.columns:
        out["notes"] = None

    out["ticker"] = out["ticker"].map(lambda x: normalize_ticker(x) if pd.notna(x) and str(x).strip() else None)
    out["sector"] = out["sector"].map(normalize_sector)
    out["is_major"] = out["is_major"].map(_coerce_bool)

    cols = ["date", "event_name", "event_type", "ticker", "sector", "is_major", "source", "notes"]
    return out[cols].sort_values(["date", "event_type", "event_name"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Event calendar construction
# -----------------------------------------------------------------------------

def build_core_event_calendar(
    ticker: str,
    market_dates: Sequence[Any],
    sector: Optional[str] = None,
    supplemental_events: Optional[pd.DataFrame] = None,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Build a canonical event table for one ticker.

    Includes:
    - next known earnings date (Yahoo)
    - FOMC meetings in the requested date range (Fed)
    - optional supplemental sector/ticker/macro events
    """
    ticker_n = normalize_ticker(ticker)
    sector_n = normalize_sector(sector)

    market_idx = pd.to_datetime(pd.Series(list(market_dates)), errors="coerce").dropna().dt.normalize().sort_values()
    if market_idx.empty:
        raise ValueError("market_dates must contain at least one valid date.")

    start_ts = market_idx.min() - pd.Timedelta(days=10)
    end_ts = market_idx.max() + pd.Timedelta(days=120)

    events: List[Dict[str, Any]] = []

    earnings_date = fetch_yahoo_earnings_date(ticker_n, session=session)
    if earnings_date is not None:
        events.append(
            EventRecord(
                date=earnings_date,
                event_name="earnings",
                event_type="earnings",
                is_major=True,
                ticker=ticker_n,
                sector=sector_n,
                source="Yahoo Finance quoteSummary/calendarEvents",
                notes="Next known earnings date",
            ).__dict__
        )

    fomc_df = fetch_fomc_calendar(start_ts, end_ts, session=session)
    if not fomc_df.empty:
        events.extend(fomc_df.to_dict(orient="records"))

    supp = supplemental_events.copy() if supplemental_events is not None else pd.DataFrame()
    if not supp.empty:
        # Apply ticker / sector / generic filtering.
        supp_ticker = supp["ticker"].fillna("").astype(str).str.upper()
        supp_sector = supp["sector"].fillna("").astype(str).str.lower()

        applies = (
            (supp_ticker == ticker_n)
            | ((supp_ticker == "") & (supp_sector == sector_n if sector_n else False))
            | ((supp_ticker == "") & (supp_sector == ""))
        )
        supp = supp[applies].copy()
        supp = supp[(supp["date"] >= start_ts) & (supp["date"] <= end_ts)].copy()
        if not supp.empty:
            events.extend(supp.to_dict(orient="records"))

    if not events:
        return pd.DataFrame(
            columns=["date", "event_name", "event_type", "is_major", "ticker", "sector", "source", "notes"]
        )

    out = pd.DataFrame(events)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["ticker"] = out["ticker"].map(lambda x: normalize_ticker(x) if pd.notna(x) and str(x).strip() else None)
    out["sector"] = out["sector"].map(normalize_sector)
    out["is_major"] = out["is_major"].map(_coerce_bool)
    out = out[out["date"].notna()].copy()
    out = out.drop_duplicates(subset=["date", "event_name", "event_type", "ticker", "sector"]).sort_values(
        ["date", "event_type", "event_name"]
    )
    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------

def _days_until_next_event(dates: pd.Series, event_dates: Sequence[pd.Timestamp]) -> pd.Series:
    event_idx = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(list(event_dates)), errors="coerce").dropna().dt.normalize().unique()))
    if len(event_idx) == 0:
        return pd.Series(np.nan, index=dates.index, dtype=float)

    values = dates.dt.normalize().to_numpy(dtype="datetime64[ns]")
    pos = event_idx.searchsorted(values, side="left")
    out = np.full(len(dates), np.nan, dtype=float)
    for i, p in enumerate(pos):
        if p < len(event_idx):
            delta = (event_idx[p] - pd.Timestamp(values[i])).days
            out[i] = float(delta)
    return pd.Series(out, index=dates.index, dtype=float)



def _flag_post_event_window(dates: pd.Series, event_dates: Sequence[pd.Timestamp], window_days: int) -> pd.Series:
    event_set = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(event_dates)), errors="coerce").dropna().dt.normalize().unique())
    if len(event_set) == 0:
        return pd.Series(0, index=dates.index, dtype=np.int8)

    result = np.zeros(len(dates), dtype=np.int8)
    norm_dates = dates.dt.normalize().tolist()
    event_lookup = set(event_set.tolist())

    # Mark t in (earnings_date, earnings_date + window_days]
    for i, dt in enumerate(norm_dates):
        for lag in range(1, window_days + 1):
            if (dt - pd.Timedelta(days=lag)) in event_lookup:
                result[i] = 1
                break
    return pd.Series(result, index=dates.index, dtype=np.int8)



def build_event_calendar_feature_frame(
    price_df: pd.DataFrame,
    ticker: str,
    sector: Optional[str] = None,
    supplemental_events: Optional[pd.DataFrame] = None,
    date_col: str = "date",
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Produce daily event-calendar features for one ticker.

    Parameters
    ----------
    price_df:
        DataFrame containing at least a date column of tradable dates.
    ticker:
        Equity ticker.
    sector:
        Optional normalized or human-readable sector name.
    supplemental_events:
        Optional DataFrame loaded from public-source catalyst calendars.
    date_col:
        Name of the date column in `price_df`.
    """
    if date_col not in price_df.columns:
        raise ValueError(f"price_df must contain '{date_col}'.")

    base = price_df[[date_col]].copy()
    base[date_col] = pd.to_datetime(base[date_col], errors="coerce").dt.normalize()
    base = base[base[date_col].notna()].sort_values(date_col).drop_duplicates().reset_index(drop=True)

    if base.empty:
        raise ValueError("price_df does not contain any valid dates.")

    events = build_core_event_calendar(
        ticker=ticker,
        market_dates=base[date_col].tolist(),
        sector=sector,
        supplemental_events=supplemental_events,
        session=session,
    )

    features = base.rename(columns={date_col: "date"}).copy()

    earnings_dates = []
    fomc_dates = []
    sector_dates = []
    major_dates = []

    if not events.empty:
        earnings_dates = events.loc[events["event_type"].eq("earnings"), "date"].tolist()
        fomc_dates = events.loc[events["event_type"].eq("fomc"), "date"].tolist()

        sector_mask = ~events["event_type"].isin(["earnings", "fomc"])
        sector_dates = events.loc[sector_mask, "date"].tolist()

        major_dates = events.loc[events["is_major"], "date"].tolist()

    features["days_to_earnings"] = _days_until_next_event(features["date"], earnings_dates)
    features["days_to_fomc"] = _days_until_next_event(features["date"], fomc_dates)
    features["days_to_sector_event"] = _days_until_next_event(features["date"], sector_dates)

    days_to_major_earnings = _days_until_next_event(features["date"], earnings_dates)
    days_to_major_fomc = _days_until_next_event(features["date"], fomc_dates)
    days_to_major_sector = _days_until_next_event(features["date"], major_dates)

    major_min = pd.concat(
        [days_to_major_earnings, days_to_major_fomc, days_to_major_sector],
        axis=1,
    ).min(axis=1, skipna=True)
    features["has_major_catalyst_30d"] = ((major_min >= 0) & (major_min <= 30)).astype(np.int8)
    features["post_earnings_5d_flag"] = _flag_post_event_window(features["date"], earnings_dates, window_days=5)

    # Diagnostics that help debugging but are optional for training.
    features["event_calendar_event_count"] = int(len(events))
    features["event_calendar_has_supplemental"] = int(
        1 if supplemental_events is not None and not supplemental_events.empty else 0
    )
    features["event_calendar_sector"] = normalize_sector(sector)
    features["event_calendar_ticker"] = normalize_ticker(ticker)

    return features



def merge_event_features(
    price_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Merge event-calendar features into a price / indicator frame."""
    if date_col not in price_df.columns:
        raise ValueError(f"price_df must contain '{date_col}'.")
    if "date" not in feature_df.columns:
        raise ValueError("feature_df must contain 'date'.")

    left = price_df.copy()
    left[date_col] = pd.to_datetime(left[date_col], errors="coerce").dt.normalize()

    right = feature_df.copy()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()

    merged = left.merge(right, how="left", left_on=date_col, right_on="date", suffixes=("", "_evt"))
    if date_col != "date":
        merged = merged.drop(columns=["date"])

    fill_zero = ["has_major_catalyst_30d", "post_earnings_5d_flag", "event_calendar_event_count", "event_calendar_has_supplemental"]
    for col in fill_zero:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(np.int8)

    # Keep forward-looking distance features as NaN when unknown. Do not invent dates.
    return merged


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lag-safe event calendar features for one ticker.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. NVDA")
    parser.add_argument("--sector", default=None, help="Optional sector label, e.g. technology, healthcare, energy")
    parser.add_argument("--price-csv", required=True, help="CSV containing at least a date column")
    parser.add_argument("--date-col", default="date", help="Date column name in the price CSV")
    parser.add_argument(
        "--supplemental-events-csv",
        default=None,
        help="Optional CSV of public-source catalyst events (sector/ticker/macro)",
    )
    parser.add_argument("--output-features", required=True, help="Path to write event feature CSV")
    parser.add_argument(
        "--output-events",
        default=None,
        help="Optional path to write the canonical event table used for feature generation",
    )
    parser.add_argument(
        "--output-merged",
        default=None,
        help="Optional path to write the original price CSV merged with event features",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG")
    return parser.parse_args()



def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    price_df = pd.read_csv(args.price_csv)
    supplemental = load_supplemental_events_csv(args.supplemental_events_csv)

    with _build_session() as session:
        feature_df = build_event_calendar_feature_frame(
            price_df=price_df,
            ticker=args.ticker,
            sector=args.sector,
            supplemental_events=supplemental,
            date_col=args.date_col,
            session=session,
        )

        feature_df.to_csv(args.output_features, index=False)
        LOGGER.info("Wrote event features to %s", args.output_features)

        if args.output_events:
            events = build_core_event_calendar(
                ticker=args.ticker,
                market_dates=pd.to_datetime(price_df[args.date_col], errors="coerce").dropna().tolist(),
                sector=args.sector,
                supplemental_events=supplemental,
                session=session,
            )
            events.to_csv(args.output_events, index=False)
            LOGGER.info("Wrote canonical event calendar to %s", args.output_events)

        if args.output_merged:
            merged = merge_event_features(price_df=price_df, feature_df=feature_df, date_col=args.date_col)
            merged.to_csv(args.output_merged, index=False)
            LOGGER.info("Wrote merged frame to %s", args.output_merged)


if __name__ == "__main__":
    main()
