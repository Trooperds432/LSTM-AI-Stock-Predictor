"""
insider_features.py
===================

Point-in-time, lag-safe insider feature engineering for equity ML pipelines.

Purpose
-------
This module implements the *insider* block of an institutional Mosaic Theory
pipeline. It pulls public SEC EDGAR ownership filings (Forms 3/4/5 and
amendments), normalizes transactions, applies compliance-aware classification,
and produces daily model features aligned by **public availability date**.

The module is intentionally opinionated:
- Public-source only.
- Uses filing availability date for lag safety.
- Separately tracks discretionary vs non-informational activity.
- Flags 10b5-1 plan activity.
- Produces exact feature columns requested by the user.

What this module does not do
----------------------------
- It does not use MNPI or infer non-public information.
- It does not perform OCR on screenshots/images. Instead, it accepts an
  optional `supplemental_transactions` DataFrame so externally parsed rows from
  filing screenshots can be merged into the calculation path.
- It does not implement macro/options/short-interest/event-calendar modules;
  those should live in separate feature builders.

Core output columns
-------------------
- ins_net_buy_usd_30d
- ins_buy_count_30d
- ins_sell_count_30d
- ins_buy_sell_ratio_30d
- ins_cluster_buy_flag
- ins_cluster_sell_flag
- ins_ceo_cfo_buy_flag
- ins_director_buy_flag
- ins_high_conviction_buy_flag
- ins_ownership_delta_pct_30d
- ins_10b5_1_sell_usd_30d
- ins_noninformational_usd_30d
- ins_first_buy_in_years_flag

Recommended usage
-----------------
1. Build / load your market-date DataFrame for the ticker.
2. Call `build_insider_feature_frame(...)`.
3. Merge on `date` into your training dataset.

SEC fair access
---------------
Pass a real SEC-compliant user agent, for example:
    "Your Name your_email@example.com"

References
----------
- SEC company tickers JSON: https://www.sec.gov/files/company_tickers.json
- SEC submissions JSON: https://data.sec.gov/submissions/
- SEC EDGAR archives: https://www.sec.gov/Archives/edgar/data/
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/"
DEFAULT_TIMEOUT = 30
DEFAULT_SLEEP_SECONDS = 0.25

FORMS_OWNERSHIP = {"3", "3/A", "4", "4/A", "5", "5/A"}
OPEN_MARKET_BUY_CODES = {"P"}
OPEN_MARKET_SELL_CODES = {"S"}
NON_INFORMATIONAL_CODES = {"A", "G", "M", "F", "C", "D", "X", "E", "I"}
HIGH_SIGNAL_ROLE_KEYWORDS = {"CEO", "CFO", "COO", "CHIEF EXECUTIVE", "CHIEF FINANCIAL", "CHIEF OPERATING"}
DIRECTOR_KEYWORDS = {"DIRECTOR", "BOARD"}

EXACT_FEATURE_COLUMNS: List[str] = [
    "ins_net_buy_usd_30d",
    "ins_buy_count_30d",
    "ins_sell_count_30d",
    "ins_buy_sell_ratio_30d",
    "ins_cluster_buy_flag",
    "ins_cluster_sell_flag",
    "ins_ceo_cfo_buy_flag",
    "ins_director_buy_flag",
    "ins_high_conviction_buy_flag",
    "ins_ownership_delta_pct_30d",
    "ins_10b5_1_sell_usd_30d",
    "ins_noninformational_usd_30d",
    "ins_first_buy_in_years_flag",
]

DIAGNOSTIC_COLUMNS: List[str] = [
    "ins_buy_value_usd_30d",
    "ins_sell_value_usd_30d",
    "ins_buy_sell_ratio_count_30d",
    "ins_discretionary_count_30d",
    "ins_discretionary_value_usd_30d",
    "ins_unique_buyers_30d",
    "ins_unique_sellers_30d",
    "ins_cluster_buy_count_30d",
    "ins_cluster_sell_count_30d",
    "ins_ceo_cfo_buy_value_usd_30d",
    "ins_director_buy_value_usd_30d",
    "ins_high_conviction_buy_value_usd_30d",
    "ins_data_freshness_days",
    "ins_source_row_count_30d",
]

ALL_OUTPUT_COLUMNS: List[str] = EXACT_FEATURE_COLUMNS + DIAGNOSTIC_COLUMNS


@dataclass(frozen=True)
class InsiderConfig:
    """Configuration for insider feature generation."""

    user_agent: str
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    lookback_days: int = 30
    baseline_90_days: int = 90
    baseline_180_days: int = 180
    years_for_first_buy_flag: int = 2
    request_sleep_seconds: float = DEFAULT_SLEEP_SECONDS
    timeout_seconds: int = DEFAULT_TIMEOUT
    include_amendments: bool = True
    max_recent_filings_to_scan: int = 400

    def __post_init__(self) -> None:
        if not self.user_agent or "@" not in self.user_agent:
            raise ValueError(
                "Provide a real SEC-compliant user_agent, e.g. 'Your Name your_email@example.com'."
            )
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if self.baseline_90_days < self.lookback_days:
            raise ValueError("baseline_90_days must be >= lookback_days")
        if self.baseline_180_days < self.baseline_90_days:
            raise ValueError("baseline_180_days must be >= baseline_90_days")


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")



def _to_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()



def _normalize_date_series(values: Iterable[Any]) -> pd.Series:
    return pd.to_datetime(pd.Series(values), errors="coerce").dt.normalize()



def make_session(user_agent: str) -> requests.Session:
    """Create a requests session with retries and SEC headers."""
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
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
            "Accept": "application/json, text/html, application/xml, text/xml, */*",
        }
    )
    return session



def _sleep(config: InsiderConfig) -> None:
    if config.request_sleep_seconds > 0:
        time.sleep(config.request_sleep_seconds)



def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        if pd.isna(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("%", "")
    text = re.sub(r"\s+", "", text)
    try:
        return float(text)
    except ValueError:
        return None



def _text_or_none(node: Optional[ET.Element]) -> Optional[str]:
    if node is None or node.text is None:
        return None
    text = re.sub(r"\s+", " ", node.text).strip()
    return text or None



def _strip_namespaces(root: ET.Element) -> ET.Element:
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]
    return root



def get_cik_for_ticker(session: requests.Session, ticker: str, config: InsiderConfig) -> int:
    """Resolve ticker to CIK using the SEC public mapping."""
    resp = session.get(SEC_TICKERS_URL, timeout=config.timeout_seconds)
    resp.raise_for_status()
    data = resp.json()
    _sleep(config)

    ticker_upper = ticker.upper().strip()
    for _, row in data.items():
        if str(row.get("ticker", "")).upper() == ticker_upper:
            return int(row["cik_str"])
    raise ValueError(f"Ticker '{ticker}' not found in SEC company_tickers.json")



def fetch_company_submissions(session: requests.Session, cik: int, config: InsiderConfig) -> Dict[str, Any]:
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    resp = session.get(url, timeout=config.timeout_seconds)
    resp.raise_for_status()
    data = resp.json()
    _sleep(config)
    return data



def _submissions_to_filings_df(submissions: Dict[str, Any], config: InsiderConfig) -> pd.DataFrame:
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()

    df = pd.DataFrame(recent)
    if df.empty:
        return df

    if "form" not in df.columns:
        return pd.DataFrame()

    df = df[df["form"].astype(str).isin(FORMS_OWNERSHIP)].copy()
    if df.empty:
        return df

    df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce").dt.normalize()
    df["acceptanceDateTime"] = pd.to_datetime(df.get("acceptanceDateTime"), errors="coerce")
    if not config.include_amendments:
        df = df[~df["form"].astype(str).str.endswith("/A")].copy()

    df = df.sort_values(["filingDate", "acceptanceDateTime"], ascending=[False, False])
    if config.max_recent_filings_to_scan:
        df = df.head(config.max_recent_filings_to_scan).copy()
    return df.reset_index(drop=True)



def _make_filing_paths(cik: int, accession_no: str) -> Tuple[str, str]:
    accession_no_nodash = accession_no.replace("-", "")
    cik_noleading = str(int(cik))
    base = f"{SEC_ARCHIVES_BASE}{cik_noleading}/{accession_no_nodash}/"
    index_json = urljoin(base, "index.json")
    return base, index_json



def fetch_filing_index(session: requests.Session, cik: int, accession_no: str, config: InsiderConfig) -> Dict[str, Any]:
    _, index_url = _make_filing_paths(cik, accession_no)
    resp = session.get(index_url, timeout=config.timeout_seconds)
    resp.raise_for_status()
    data = resp.json()
    _sleep(config)
    return data



def _choose_primary_ownership_xml(index_json: Dict[str, Any]) -> Optional[str]:
    directory = index_json.get("directory", {})
    items = directory.get("item", [])
    if not items:
        return None

    candidates: List[str] = []
    for item in items:
        name = str(item.get("name", ""))
        lower = name.lower()
        if not lower.endswith(".xml"):
            continue
        if "ownership" in lower or "xslF345X" in lower or lower in {"primary_doc.xml", "form4.xml", "form5.xml", "form3.xml"}:
            candidates.append(name)

    if not candidates:
        for item in items:
            name = str(item.get("name", ""))
            if name.lower().endswith(".xml"):
                candidates.append(name)

    return candidates[0] if candidates else None



def fetch_filing_xml(session: requests.Session, cik: int, accession_no: str, config: InsiderConfig) -> Tuple[str, str]:
    base, _ = _make_filing_paths(cik, accession_no)
    index_json = fetch_filing_index(session, cik, accession_no, config)
    xml_name = _choose_primary_ownership_xml(index_json)
    if not xml_name:
        raise FileNotFoundError(f"No XML found for accession {accession_no}")
    xml_url = urljoin(base, xml_name)
    resp = session.get(xml_url, timeout=config.timeout_seconds)
    resp.raise_for_status()
    _sleep(config)
    return xml_url, resp.text



def _find_text(root: ET.Element, path: str) -> Optional[str]:
    node = root.find(path)
    return _text_or_none(node)



def _extract_footnotes(root: ET.Element) -> Dict[str, str]:
    notes: Dict[str, str] = {}
    for footnote in root.findall("footnotes/footnote"):
        fid = footnote.attrib.get("id")
        text = _text_or_none(footnote)
        if fid and text:
            notes[fid] = text
    return notes



def _collect_footnote_ids(node: ET.Element) -> List[str]:
    ids: List[str] = []
    for child in node.iter():
        if child.tag == "footnoteId":
            fid = child.attrib.get("id")
            if fid:
                ids.append(fid)
    return ids



def _detect_10b5_1(notes_text: str) -> bool:
    text = notes_text.lower()
    patterns = [
        "10b5-1",
        "10b5 1",
        "rule 10b5-1",
        "10b5-1 trading plan",
        "trading plan adopted",
        "adopted pursuant to rule 10b5-1",
    ]
    return any(p in text for p in patterns)



def _detect_tax_withholding(notes_text: str) -> bool:
    text = notes_text.lower()
    return any(
        p in text
        for p in [
            "tax withholding",
            "withholding tax",
            "to satisfy tax",
            "sell-to-cover",
            "sell to cover",
            "withheld for taxes",
        ]
    )



def _detect_routine_grant(notes_text: str) -> bool:
    text = notes_text.lower()
    return any(
        p in text
        for p in [
            "restricted stock unit",
            "rsu",
            "option exercise",
            "stock option",
            "vesting",
            "vested",
            "grant",
            "award",
            "performance share",
            "deferred stock unit",
        ]
    )



def _detect_gift(notes_text: str) -> bool:
    return "gift" in notes_text.lower()



def _normalize_role(is_director: Optional[str], is_officer: Optional[str], officer_title: Optional[str], is_ten_owner: Optional[str], is_other: Optional[str], other_text: Optional[str]) -> str:
    parts: List[str] = []
    if str(is_director).strip() == "1":
        parts.append("Director")
    if str(is_officer).strip() == "1":
        title = officer_title or "Officer"
        parts.append(title)
    if str(is_ten_owner).strip() == "1":
        parts.append(">10% Owner")
    if str(is_other).strip() == "1":
        parts.append(other_text or "Other")
    return " | ".join(parts) if parts else "Unknown"



def _role_flags(role: str) -> Dict[str, bool]:
    role_upper = role.upper()
    is_ceo_cfo_coo = any(keyword in role_upper for keyword in HIGH_SIGNAL_ROLE_KEYWORDS)
    is_director = any(keyword in role_upper for keyword in DIRECTOR_KEYWORDS)
    is_ten_owner = ">10% OWNER" in role_upper
    return {
        "is_ceo_cfo_coo": is_ceo_cfo_coo,
        "is_director": is_director,
        "is_ten_owner": is_ten_owner,
    }



def _classify_transaction(raw_code: Optional[str], notes_text: str) -> Dict[str, Any]:
    code = (raw_code or "").strip().upper()
    notes = notes_text.lower()

    is_10b5_1 = _detect_10b5_1(notes_text)
    is_tax = _detect_tax_withholding(notes_text)
    is_routine_grant = _detect_routine_grant(notes_text)
    is_gift = _detect_gift(notes_text) or code == "G"

    if code in OPEN_MARKET_BUY_CODES:
        bucket = "P"
        informational = True
        direction = "buy"
        note_type = "open_market_buy"
    elif code in OPEN_MARKET_SELL_CODES:
        bucket = "S"
        informational = not is_10b5_1
        direction = "sell"
        note_type = "open_market_sale_10b5_1" if is_10b5_1 else "open_market_sale"
    elif code in {"M", "A"}:
        bucket = code
        informational = False
        direction = "other"
        note_type = "option_or_award"
    elif is_gift:
        bucket = "G"
        informational = False
        direction = "other"
        note_type = "gift"
    elif code in NON_INFORMATIONAL_CODES or is_tax or is_routine_grant:
        bucket = code or "OTHER"
        informational = False
        direction = "other"
        note_type = "non_informational"
    else:
        bucket = code or "OTHER"
        informational = False
        direction = "other"
        note_type = "other"

    routine = (not informational) or is_10b5_1 or is_tax or is_routine_grant or is_gift
    return {
        "code_bucket": bucket,
        "direction": direction,
        "is_10b5_1": bool(is_10b5_1),
        "is_tax_withholding": bool(is_tax),
        "is_gift": bool(is_gift),
        "is_routine_grant": bool(is_routine_grant),
        "is_informational": bool(informational),
        "is_non_informational": bool(not informational),
        "routine_vs_discretionary": "routine" if routine else "discretionary",
        "notes_category": note_type,
    }



def parse_ownership_xml(
    xml_text: str,
    ticker: str,
    cik: int,
    filing_accession_no: str,
    filing_date: pd.Timestamp,
    filing_acceptance_dt: Optional[pd.Timestamp],
    filing_form: str,
    source_url: str,
) -> pd.DataFrame:
    """
    Parse a Form 3/4/5 ownership XML into a normalized transaction table.

    Returns one row per transaction line item.
    """
    root = ET.fromstring(xml_text)
    root = _strip_namespaces(root)
    footnotes = _extract_footnotes(root)

    issuer_ticker = _find_text(root, "issuer/issuerTradingSymbol") or ticker.upper()
    issuer_name = _find_text(root, "issuer/issuerName")

    rows: List[Dict[str, Any]] = []

    for reporting_owner in root.findall("reportingOwner"):
        owner_name = _find_text(reporting_owner, "reportingOwnerId/rptOwnerName")
        owner_cik = _find_text(reporting_owner, "reportingOwnerId/rptOwnerCik")
        owner_rel = reporting_owner.find("reportingOwnerRelationship")
        if owner_rel is not None:
            role = _normalize_role(
                _find_text(owner_rel, "isDirector"),
                _find_text(owner_rel, "isOfficer"),
                _find_text(owner_rel, "officerTitle"),
                _find_text(owner_rel, "isTenPercentOwner"),
                _find_text(owner_rel, "isOther"),
                _find_text(owner_rel, "otherText"),
            )
        else:
            role = "Unknown"

        role_info = _role_flags(role)

        for table_name, transaction_kind in [
            ("nonDerivativeTable", "non_derivative"),
            ("derivativeTable", "derivative"),
        ]:
            table = reporting_owner.find(f"../{table_name}")  # not reliable for nested owner loops
            if table is None:
                table = root.find(table_name)
            if table is None:
                continue

            for txn in table.findall("nonDerivativeTransaction") + table.findall("derivativeTransaction"):
                security_title = _find_text(txn, "securityTitle/value")
                transaction_date = _find_text(txn, "transactionDate/value")
                deemed_date = _find_text(txn, "deemedExecutionDate/value")
                raw_code = _find_text(txn, "transactionCoding/transactionCode")
                shares = _safe_float(_find_text(txn, "transactionAmounts/transactionShares/value"))
                price = _safe_float(_find_text(txn, "transactionAmounts/transactionPricePerShare/value"))
                acquired_disposed = _find_text(txn, "transactionAmounts/transactionAcquiredDisposedCode/value")
                shares_following = _safe_float(_find_text(txn, "postTransactionAmounts/sharesOwnedFollowingTransaction/value"))
                ownership_nature = _find_text(txn, "ownershipNature/directOrIndirectOwnership/value")

                footnote_ids = _collect_footnote_ids(txn)
                notes_text = " | ".join(footnotes[fid] for fid in footnote_ids if fid in footnotes)
                class_info = _classify_transaction(raw_code, notes_text)

                tx_date = pd.to_datetime(transaction_date or deemed_date, errors="coerce")
                tx_date = tx_date.normalize() if not pd.isna(tx_date) else pd.NaT

                value_usd: Optional[float]
                if shares is not None and price is not None:
                    value_usd = float(shares) * float(price)
                else:
                    value_usd = None

                signed_shares: Optional[float] = shares
                signed_value_usd: Optional[float] = value_usd
                if class_info["direction"] == "sell":
                    signed_shares = -shares if shares is not None else None
                    signed_value_usd = -value_usd if value_usd is not None else None

                pct_ownership_delta: Optional[float] = None
                pre_shares_est: Optional[float] = None
                if shares_following is not None and shares is not None:
                    if class_info["direction"] == "buy":
                        pre_shares_est = shares_following - shares
                        delta_shares = shares
                    elif class_info["direction"] == "sell":
                        pre_shares_est = shares_following + shares
                        delta_shares = -shares
                    else:
                        delta_shares = 0.0
                    if pre_shares_est is not None and pre_shares_est > 0:
                        pct_ownership_delta = 100.0 * float(delta_shares) / float(pre_shares_est)

                availability_date = filing_date.normalize()
                citation = source_url

                row: Dict[str, Any] = {
                    "ticker": issuer_ticker,
                    "issuer_name": issuer_name,
                    "issuer_cik": int(cik),
                    "filing_accession_no": filing_accession_no,
                    "filing_form": filing_form,
                    "filing_date": filing_date.normalize(),
                    "filing_acceptance_dt": filing_acceptance_dt,
                    "availability_date": availability_date,
                    "trade_date": tx_date,
                    "insider": owner_name,
                    "insider_cik": owner_cik,
                    "role": role,
                    "security_title": security_title,
                    "transaction_kind": transaction_kind,
                    "raw_code": raw_code,
                    "shares": shares,
                    "price": price,
                    "value_usd": value_usd,
                    "signed_shares": signed_shares,
                    "signed_value_usd": signed_value_usd,
                    "shares_following": shares_following,
                    "pre_shares_est": pre_shares_est,
                    "ownership_nature": ownership_nature,
                    "pct_ownership_delta": pct_ownership_delta,
                    "transaction_acquired_disposed_code": acquired_disposed,
                    "notes_text": notes_text,
                    "source_url": source_url,
                    "source_type": "SEC_EDGAR",
                    "citation": citation,
                    "data_timestamp_utc": _utc_now(),
                    **class_info,
                    **role_info,
                }
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return normalize_transaction_table(df)



def normalize_transaction_table(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize types and fill derived helper columns."""
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "ticker", "filing_date", "availability_date", "trade_date", "insider", "role",
            "raw_code", "code_bucket", "shares", "price", "value_usd", "signed_shares",
            "signed_value_usd", "shares_following", "pct_ownership_delta", "is_10b5_1",
            "is_informational", "is_non_informational", "routine_vs_discretionary",
            "source_url", "citation", "data_timestamp_utc"
        ])

    out = df.copy()

    date_cols = ["filing_date", "availability_date", "trade_date"]
    for col in date_cols:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()

    if "filing_acceptance_dt" in out.columns:
        out["filing_acceptance_dt"] = pd.to_datetime(out["filing_acceptance_dt"], errors="coerce")
    if "data_timestamp_utc" in out.columns:
        out["data_timestamp_utc"] = pd.to_datetime(out["data_timestamp_utc"], errors="coerce", utc=True)
    else:
        out["data_timestamp_utc"] = _utc_now()

    text_cols = [
        "ticker", "issuer_name", "filing_accession_no", "filing_form", "insider", "insider_cik",
        "role", "security_title", "transaction_kind", "raw_code", "code_bucket", "notes_text",
        "source_url", "source_type", "citation", "ownership_nature", "notes_category",
        "routine_vs_discretionary",
    ]
    for col in text_cols:
        if col not in out.columns:
            out[col] = None
        out[col] = out[col].astype("string")

    num_cols = [
        "shares", "price", "value_usd", "signed_shares", "signed_value_usd", "shares_following",
        "pre_shares_est", "pct_ownership_delta", "issuer_cik",
    ]
    for col in num_cols:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    bool_cols = [
        "is_10b5_1", "is_tax_withholding", "is_gift", "is_routine_grant", "is_informational",
        "is_non_informational", "is_ceo_cfo_coo", "is_director", "is_ten_owner",
    ]
    for col in bool_cols:
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].fillna(False).astype(bool)

    # Secondary helper flags.
    out["is_buy"] = out["direction"].eq("buy")
    out["is_sell"] = out["direction"].eq("sell")
    out["is_discretionary_buy"] = out["is_buy"] & out["is_informational"]
    out["is_discretionary_sell"] = out["is_sell"] & out["is_informational"] & (~out["is_10b5_1"])
    out["is_discretionary_trade"] = out["is_discretionary_buy"] | out["is_discretionary_sell"]
    out["is_noninformational_line"] = out["is_non_informational"] | out["raw_code"].isin(list(NON_INFORMATIONAL_CODES))

    out = out.sort_values(["availability_date", "trade_date", "filing_date", "insider", "raw_code"], kind="stable")
    return out.reset_index(drop=True)



def fetch_insider_transactions(
    ticker: str,
    config: InsiderConfig,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Pull and parse SEC ownership filings for the ticker.

    The parser scans recent Forms 3/4/5 and amendments from the company's
    submissions feed, then downloads the primary XML for each filing.
    """
    own_session = session or make_session(config.user_agent)
    cik = get_cik_for_ticker(own_session, ticker, config)
    submissions = fetch_company_submissions(own_session, cik, config)
    filings_df = _submissions_to_filings_df(submissions, config)
    if filings_df.empty:
        return pd.DataFrame()

    earliest_needed = None
    if config.start_date is not None:
        earliest_needed = _to_ts(config.start_date) - pd.Timedelta(days=max(config.baseline_180_days, config.lookback_days) + 30)

    rows: List[pd.DataFrame] = []
    for filing in filings_df.to_dict(orient="records"):
        filing_date = pd.to_datetime(filing.get("filingDate"), errors="coerce")
        if pd.isna(filing_date):
            continue
        filing_date = filing_date.normalize()

        if earliest_needed is not None and filing_date < earliest_needed:
            continue

        accession_no = str(filing.get("accessionNumber"))
        filing_form = str(filing.get("form"))
        acceptance_dt = pd.to_datetime(filing.get("acceptanceDateTime"), errors="coerce")

        try:
            xml_url, xml_text = fetch_filing_xml(own_session, cik, accession_no, config)
            parsed = parse_ownership_xml(
                xml_text=xml_text,
                ticker=ticker,
                cik=cik,
                filing_accession_no=accession_no,
                filing_date=filing_date,
                filing_acceptance_dt=acceptance_dt if not pd.isna(acceptance_dt) else None,
                filing_form=filing_form,
                source_url=xml_url,
            )
            if not parsed.empty:
                rows.append(parsed)
        except Exception as exc:  # noqa: BLE001 - best-effort pipeline
            LOGGER.warning("Failed parsing %s %s for %s: %s", filing_form, accession_no, ticker, exc)
            continue

    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    return normalize_transaction_table(combined)



def merge_supplemental_transactions(
    sec_transactions: pd.DataFrame,
    supplemental_transactions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge externally parsed transaction rows, e.g. from filing screenshots.

    This is the safe compromise for attachment handling: do image parsing
    upstream, convert it into rows with the normalized schema, then merge here.
    """
    if supplemental_transactions is None or supplemental_transactions.empty:
        return normalize_transaction_table(sec_transactions)

    supp = normalize_transaction_table(supplemental_transactions)
    base = normalize_transaction_table(sec_transactions)

    merged = pd.concat([base, supp], ignore_index=True, sort=False)

    dedupe_cols = [
        "ticker", "filing_accession_no", "insider", "trade_date", "filing_date",
        "raw_code", "shares", "price", "value_usd", "source_url"
    ]
    dedupe_cols = [c for c in dedupe_cols if c in merged.columns]
    if dedupe_cols:
        merged = merged.drop_duplicates(subset=dedupe_cols, keep="first")

    return normalize_transaction_table(merged)



def _ensure_date_index(price_or_dates: pd.DataFrame | pd.Series | Sequence[Any], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Create the market-date frame used for feature alignment.

    Accepts:
    - DataFrame with a `date` column
    - Series/list-like of dates
    - None-like -> business-date range
    """
    if isinstance(price_or_dates, pd.DataFrame):
        if "date" in price_or_dates.columns:
            out = price_or_dates[["date"]].copy()
        else:
            out = pd.DataFrame({"date": price_or_dates.index})
    elif isinstance(price_or_dates, pd.Series):
        out = pd.DataFrame({"date": price_or_dates.values})
    else:
        out = pd.DataFrame({"date": list(price_or_dates)}) if price_or_dates is not None else pd.DataFrame()

    if out.empty:
        out = pd.DataFrame({"date": pd.bdate_range(start_date, end_date)})

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"]).drop_duplicates().sort_values("date").reset_index(drop=True)
    return out



def _rolling_window_slice(df: pd.DataFrame, as_of: pd.Timestamp, days: int, date_col: str = "availability_date") -> pd.DataFrame:
    lower = as_of - pd.Timedelta(days=days - 1)
    return df[(df[date_col] >= lower) & (df[date_col] <= as_of)].copy()



def _week_key(ts: pd.Timestamp) -> Tuple[int, int]:
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)



def _safe_ratio(numerator: float, denominator: float, default_if_zero: float = 0.0) -> float:
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return float(default_if_zero)
    return float(numerator) / float(denominator)



def _infer_first_buy_flags(transactions: pd.DataFrame, years: int = 2) -> pd.DataFrame:
    """Flag open-market buys where the insider has no prior buy in the lookback years."""
    if transactions.empty:
        out = transactions.copy()
        out["is_first_buy_in_years"] = False
        return out

    out = transactions.copy()
    out["is_first_buy_in_years"] = False

    buys = out[out["is_discretionary_buy"]].copy()
    if buys.empty:
        return out

    lookback_days = int(round(years * 365.25))
    for insider, g in buys.groupby("insider", sort=False):
        g = g.sort_values("availability_date")
        prior_dates: List[pd.Timestamp] = []
        idxs: List[int] = []
        flags: List[bool] = []
        for idx, row in g.iterrows():
            as_of = row["availability_date"]
            cutoff = as_of - pd.Timedelta(days=lookback_days)
            prior_recent = [d for d in prior_dates if d >= cutoff]
            flags.append(len(prior_recent) == 0)
            idxs.append(idx)
            prior_dates.append(as_of)
        out.loc[idxs, "is_first_buy_in_years"] = flags
    return out



def _compute_window_metrics(window_df: pd.DataFrame) -> Dict[str, float]:
    discretionary_buys = window_df[window_df["is_discretionary_buy"]].copy()
    discretionary_sells = window_df[window_df["is_discretionary_sell"]].copy()
    noninformational = window_df[window_df["is_noninformational_line"]].copy()
    tenb5_sells = window_df[window_df["is_sell"] & window_df["is_10b5_1"]].copy()

    buy_value = float(discretionary_buys["value_usd"].fillna(0.0).sum())
    sell_value = float(discretionary_sells["value_usd"].fillna(0.0).sum())
    net_buy_usd = buy_value - sell_value

    buy_count = int(len(discretionary_buys))
    sell_count = int(len(discretionary_sells))

    buy_sell_ratio_usd = _safe_ratio(buy_value, sell_value, default_if_zero=(np.inf if buy_value > 0 else 0.0))
    buy_sell_ratio_count = _safe_ratio(buy_count, sell_count, default_if_zero=(np.inf if buy_count > 0 else 0.0))

    # Cluster events: >=2 unique insiders trading same direction in the same ISO week.
    cluster_buy_count = 0
    cluster_sell_count = 0
    if not discretionary_buys.empty:
        by_week = discretionary_buys.groupby(discretionary_buys["availability_date"].map(_week_key))["insider"].nunique()
        cluster_buy_count = int((by_week >= 2).sum())
    if not discretionary_sells.empty:
        by_week = discretionary_sells.groupby(discretionary_sells["availability_date"].map(_week_key))["insider"].nunique()
        cluster_sell_count = int((by_week >= 2).sum())

    ceo_cfo_buys = discretionary_buys[discretionary_buys["is_ceo_cfo_coo"]]
    director_buys = discretionary_buys[discretionary_buys["is_director"]]

    # High-conviction rules from the spec.
    high_conviction_mask = (
        discretionary_buys["value_usd"].fillna(0.0).ge(1_000_000.0)
        | (discretionary_buys["is_ceo_cfo_coo"] & discretionary_buys["pct_ownership_delta"].fillna(0.0).ge(1.0))
        | discretionary_buys.get("is_first_buy_in_years", pd.Series(False, index=discretionary_buys.index)).fillna(False)
    )
    high_conviction_buys = discretionary_buys[high_conviction_mask]

    metrics: Dict[str, float] = {
        "ins_net_buy_usd_30d": net_buy_usd,
        "ins_buy_count_30d": float(buy_count),
        "ins_sell_count_30d": float(sell_count),
        "ins_buy_sell_ratio_30d": float(buy_sell_ratio_usd),
        "ins_cluster_buy_flag": float(cluster_buy_count > 0),
        "ins_cluster_sell_flag": float(cluster_sell_count > 0),
        "ins_ceo_cfo_buy_flag": float(not ceo_cfo_buys.empty),
        "ins_director_buy_flag": float(not director_buys.empty),
        "ins_high_conviction_buy_flag": float(not high_conviction_buys.empty),
        "ins_ownership_delta_pct_30d": float(discretionary_buys["pct_ownership_delta"].fillna(0.0).sum()),
        "ins_10b5_1_sell_usd_30d": float(tenb5_sells["value_usd"].fillna(0.0).sum()),
        "ins_noninformational_usd_30d": float(noninformational["value_usd"].fillna(0.0).sum()),
        "ins_first_buy_in_years_flag": float(
            discretionary_buys.get("is_first_buy_in_years", pd.Series(False, index=discretionary_buys.index)).fillna(False).any()
        ),
        # Diagnostics.
        "ins_buy_value_usd_30d": buy_value,
        "ins_sell_value_usd_30d": sell_value,
        "ins_buy_sell_ratio_count_30d": float(buy_sell_ratio_count),
        "ins_discretionary_count_30d": float(len(window_df[window_df["is_discretionary_trade"]])),
        "ins_discretionary_value_usd_30d": float(window_df.loc[window_df["is_discretionary_trade"], "value_usd"].fillna(0.0).sum()),
        "ins_unique_buyers_30d": float(discretionary_buys["insider"].nunique()),
        "ins_unique_sellers_30d": float(discretionary_sells["insider"].nunique()),
        "ins_cluster_buy_count_30d": float(cluster_buy_count),
        "ins_cluster_sell_count_30d": float(cluster_sell_count),
        "ins_ceo_cfo_buy_value_usd_30d": float(ceo_cfo_buys["value_usd"].fillna(0.0).sum()),
        "ins_director_buy_value_usd_30d": float(director_buys["value_usd"].fillna(0.0).sum()),
        "ins_high_conviction_buy_value_usd_30d": float(high_conviction_buys["value_usd"].fillna(0.0).sum()),
        "ins_source_row_count_30d": float(len(window_df)),
    }
    return metrics



def build_daily_insider_features(
    transactions: pd.DataFrame,
    market_dates: pd.DataFrame,
    config: InsiderConfig,
) -> pd.DataFrame:
    """
    Build lag-safe daily insider features aligned to market dates.

    Parameters
    ----------
    transactions:
        Normalized transaction table. Must already be public-source only.
    market_dates:
        DataFrame with a `date` column to align the features to.
    config:
        Insider feature configuration.
    """
    dates = market_dates.copy()
    dates["date"] = pd.to_datetime(dates["date"], errors="coerce").dt.normalize()
    dates = dates.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if transactions is None or transactions.empty:
        out = dates.copy()
        for col in ALL_OUTPUT_COLUMNS:
            out[col] = 0.0
        return out

    tx = normalize_transaction_table(transactions)
    tx = _infer_first_buy_flags(tx, years=config.years_for_first_buy_flag)

    out_rows: List[Dict[str, Any]] = []
    max_availability = tx["availability_date"].max()

    for as_of in dates["date"]:
        window_30d = _rolling_window_slice(tx, as_of, config.lookback_days, date_col="availability_date")
        metrics = _compute_window_metrics(window_30d)
        freshness_days = np.nan
        if pd.notna(max_availability):
            freshness_days = float((as_of - max_availability).days)
            freshness_days = max(freshness_days, 0.0)

        row = {"date": as_of, **metrics, "ins_data_freshness_days": freshness_days}
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    for col in ALL_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    out = out[["date"] + ALL_OUTPUT_COLUMNS]
    return out



def summarize_insider_window(
    transactions: pd.DataFrame,
    as_of_date: Any,
    lookback_days: int = 30,
) -> Dict[str, Any]:
    """Produce a report-style summary for the trailing insider window."""
    if transactions is None or transactions.empty:
        return {
            "as_of_date": _to_ts(as_of_date),
            "lookback_days": lookback_days,
            "transactions": 0,
            "summary": {},
            "top_rows": [],
            "limitations": ["No SEC ownership filings parsed for the requested window."],
        }

    tx = normalize_transaction_table(transactions)
    tx = _infer_first_buy_flags(tx)
    as_of = _to_ts(as_of_date)
    window_df = _rolling_window_slice(tx, as_of, lookback_days, date_col="availability_date")
    metrics = _compute_window_metrics(window_df)

    top_rows_df = window_df.sort_values(["availability_date", "value_usd"], ascending=[False, False]).head(20).copy()
    top_rows = top_rows_df[
        [
            "trade_date",
            "filing_date",
            "insider",
            "role",
            "code_bucket",
            "shares",
            "price",
            "value_usd",
            "shares_following",
            "is_10b5_1",
            "routine_vs_discretionary",
            "citation",
        ]
    ].to_dict(orient="records")

    limitations: List[str] = []
    if top_rows_df.empty:
        limitations.append("No qualifying transactions in the trailing window.")
    if window_df["trade_date"].isna().any():
        limitations.append("Some filings lacked explicit trade dates; filing date alignment was used for availability only.")
    if window_df["value_usd"].isna().any():
        limitations.append("Some line items omitted price or value; affected rows were still retained for categorical features.")

    return {
        "as_of_date": as_of,
        "lookback_days": lookback_days,
        "transactions": int(len(window_df)),
        "summary": metrics,
        "top_rows": top_rows,
        "limitations": limitations,
    }



def build_insider_feature_frame(
    ticker: str,
    price_or_dates: Optional[pd.DataFrame | pd.Series | Sequence[Any]] = None,
    config: Optional[InsiderConfig] = None,
    supplemental_transactions: Optional[pd.DataFrame] = None,
    session: Optional[requests.Session] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end helper: fetch, normalize, merge supplementals, and build features.

    Returns
    -------
    features_df, transactions_df
    """
    if config is None:
        raise ValueError("config is required")

    end_date = _to_ts(config.end_date or pd.Timestamp.today())
    start_date = _to_ts(
        config.start_date or (end_date - pd.Timedelta(days=max(config.baseline_180_days, config.lookback_days) + 30))
    )

    market_dates = _ensure_date_index(price_or_dates, start_date, end_date)
    transactions = fetch_insider_transactions(ticker=ticker, config=config, session=session)
    transactions = merge_supplemental_transactions(transactions, supplemental_transactions=supplemental_transactions)

    if not transactions.empty:
        transactions = transactions[transactions["availability_date"].notna()].copy()
        transactions = transactions[transactions["availability_date"] <= end_date].copy()
        transactions = transactions[transactions["availability_date"] >= (start_date - pd.Timedelta(days=max(config.baseline_180_days, config.lookback_days) + 30))].copy()

    features = build_daily_insider_features(transactions=transactions, market_dates=market_dates, config=config)
    return features, transactions



def merge_into_price_frame(price_df: pd.DataFrame, feature_df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    """Convenience helper to merge insider features into a price DataFrame."""
    if "date" not in price_df.columns:
        raise ValueError("price_df must contain a 'date' column")
    out = price_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    feat = feature_df.copy()
    feat["date"] = pd.to_datetime(feat["date"], errors="coerce").dt.normalize()
    merged = out.merge(feat, on="date", how="left")
    for col in ALL_OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = fill_value
        merged[col] = merged[col].replace([np.inf, -np.inf], np.nan).fillna(fill_value)
    return merged



def load_price_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        # accept index-like first column names commonly used in exported market data
        first_col = df.columns[0]
        if first_col.lower() in {"date", "datetime", "timestamp"}:
            df = df.rename(columns={first_col: "date"})
        else:
            raise ValueError("CSV must contain a 'date' column")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)



def load_supplemental_transactions_csv(path: str) -> pd.DataFrame:
    """
    Load externally parsed filing rows from CSV.

    Expected columns are the normalized transaction schema or a subset of it.
    At minimum, provide dates plus identity/value fields when available.
    """
    df = pd.read_csv(path)
    return normalize_transaction_table(df)



def main() -> None:
    parser = argparse.ArgumentParser(description="Build lag-safe insider features from SEC ownership filings.")
    parser.add_argument("--ticker", required=True, help="Equity ticker, e.g. NVDA")
    parser.add_argument("--user-agent", required=True, help="SEC-compliant user agent, e.g. 'Your Name your_email@example.com'")
    parser.add_argument("--price-csv", help="Optional CSV with a date column to align features to market dates")
    parser.add_argument("--supplemental-transactions-csv", help="Optional CSV of externally parsed filing image rows")
    parser.add_argument("--start-date", help="Optional feature start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="Optional feature end date YYYY-MM-DD")
    parser.add_argument("--output-features", default="insider_features_out.csv", help="Output CSV path for daily features")
    parser.add_argument("--output-transactions", default="insider_transactions_out.csv", help="Output CSV path for normalized transactions")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    price_df = load_price_csv(args.price_csv) if args.price_csv else None
    supp_df = load_supplemental_transactions_csv(args.supplemental_transactions_csv) if args.supplemental_transactions_csv else None

    config = InsiderConfig(
        user_agent=args.user_agent,
        start_date=pd.Timestamp(args.start_date) if args.start_date else None,
        end_date=pd.Timestamp(args.end_date) if args.end_date else None,
    )

    features_df, tx_df = build_insider_feature_frame(
        ticker=args.ticker,
        price_or_dates=price_df,
        config=config,
        supplemental_transactions=supp_df,
    )

    features_df.to_csv(args.output_features, index=False)
    tx_df.to_csv(args.output_transactions, index=False)
    print(f"Wrote {len(features_df):,} feature rows to {args.output_features}")
    print(f"Wrote {len(tx_df):,} transaction rows to {args.output_transactions}")


if __name__ == "__main__":
    main()
