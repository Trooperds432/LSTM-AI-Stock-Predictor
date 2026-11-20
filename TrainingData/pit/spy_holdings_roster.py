"""Independent S&P 500 roster around 2010-12-31 from SEC fund holdings filings.

Why three filings
-----------------
SPDR S&P 500 ETF Trust (SPY, CIK 884394) is a unit investment trust. It files no Form N-Q.
Its only full holdings schedules on EDGAR are the semiannual N-30D shareholder reports, dated
2010-09-30 (annual) and 2011-03-31 (semiannual). No SPY schedule is dated 2010-12-31.
The exact-date full schedule of an S&P 500 index fund is iShares Trust's Form N-Q for period
2010-12-31, which includes the iShares S&P 500 Index Fund (IVV). IVV is therefore the primary
exact-date source, and the two SPY reports bracket it.

Identity links (strongest first; nothing ambiguous is guessed)
--------------------------------------------------------------
The schedules print issuer names but no CUSIPs, and the replay holds LSEG ISINs, which are
*current* identifiers for surviving companies (for example Exxon is US30233Q1085, not its 2010
ISIN US30231G1022). Each holding line is linked to a replay ISIN by:

1. ``cusip_issuer``: holding name -> 6-character CUSIP issuer number via the SEC "Official List
   of Section 13(f) Securities", Q4 2010 (current as of 2010-12-15), equal to the issuer number
   embedded in a CUSIP-based replay ISIN.
2. ``lseg_name``: for lines still unlinked, holding name equal to an LSEG constituent or common
   name of a replay ISIN (from the LSEG snapshot files), one-to-one only.
Every link reports the price ratio LSEG close / (value / shares) as corroboration. Holdings are
valued at the report date's close, and the LSEG prices are unadjusted.
3. ``price_candidate``: unlinked pairs whose prices agree within tolerance, one-to-one. These
   are listed for manual review and are *not* counted as matches.

Usage (from repo root):
    python TrainingData/pit/spy_holdings_roster.py            # download (cached), parse, compare
    python TrainingData/pit/spy_holdings_roster.py --offline  # cached files only

Set SEC_USER_AGENT to override the default generic EDGAR User-Agent.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEC_CACHE = ROOT / "TrainingData" / "indicators_data" / "sec"
LSEG_ROOT = ROOT / "TrainingData" / "indicators_data" / "lseg"
LSEG_DIR = LSEG_ROOT / "20261119T132813190380Z"
OUT_CSV = ROOT / "TrainingData" / "pit" / "overrides" / "spy_holdings_2010-12-31.csv"
OUT_JSON = ROOT / "research" / "spy_roster_2010_comparison.json"
OUT_MD = ROOT / "research" / "spy_roster_2010_comparison.md"

DEFAULT_UA = "LSTM-research-bot admin@example.com"
OPENING_SESSION = "2011-01-03"
PRICE_TOL = 0.05  # |log(LSEG close / implied price)| accepted as corroboration


@dataclass(frozen=True)
class Source:
    key: str
    fund: str
    cik: int
    accession: str
    document: str
    as_of: str
    form: str
    role: str
    fund_title: str | None = None  # page title to keep in multi-fund filings

    @property
    def url(self) -> str:
        return (f"https://www.sec.gov/Archives/edgar/data/{self.cik}/"
                f"{self.accession.replace('-', '')}/{self.document}")

    @property
    def index_url(self) -> str:
        return (f"https://www.sec.gov/Archives/edgar/data/{self.cik}/"
                f"{self.accession.replace('-', '')}/{self.accession}-index.htm")


SOURCES = [
    Source("ivv_2010-12-31", "iShares S&P 500 Index Fund (IVV), iShares Trust", 1100663,
           "0001193125-11-052046", "dnq.htm", "2010-12-31", "N-Q", "primary_exact_date",
           fund_title="S&P 500 INDEX FUND"),
    Source("spy_2010-09-30", "SPDR S&P 500 ETF Trust (SPY)", 884394,
           "0000950123-10-109631", "b82994a1nv30d.htm", "2010-09-30", "N-30D", "bracket_before"),
    Source("spy_2011-03-31", "SPDR S&P 500 ETF Trust (SPY)", 884394,
           "0000950123-11-054920", "b85930a1nv30d.htm", "2011-03-31", "N-30D", "bracket_after"),
]
# (source key, replay session, LSEG price session or None)
COMPARISONS = [
    ("ivv_2010-12-31", OPENING_SESSION, OPENING_SESSION),
    ("spy_2011-03-31", "2011-03-31", "2011-03-31"),
    ("spy_2010-09-30", OPENING_SESSION, None),  # reference only: one quarter before the replay
]
THIRTEENF_URL = "https://www.sec.gov/divisions/investment/13f/13flist2010q4.pdf"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK0000884394.json"


# --------------------------------------------------------------------------- identifiers
def _char_value(c: str) -> int:
    if c.isdigit():
        return int(c)
    if c.isalpha():
        return ord(c.upper()) - 55  # A=10 .. Z=35
    return {"*": 36, "@": 37, "#": 38}[c]


def cusip_check_digit(cusip8: str) -> int:
    if len(cusip8) != 8:
        raise ValueError(f"need 8 characters, got {cusip8!r}")
    total = 0
    for i, c in enumerate(cusip8):
        v = _char_value(c)
        if i % 2 == 1:
            v *= 2
        total += v // 10 + v % 10
    return (10 - total % 10) % 10


def is_valid_cusip(cusip: str) -> bool:
    return (len(cusip) == 9 and cusip[8].isdigit()
            and all(c.isalnum() or c in "*@#" for c in cusip[:8])
            and cusip_check_digit(cusip[:8]) == int(cusip[8]))


def isin_check_digit(body11: str) -> int:
    if len(body11) != 11 or not body11.isalnum():
        raise ValueError(f"need 11 alphanumerics, got {body11!r}")
    digits = "".join(str(_char_value(c)) for c in body11.upper())
    total = 0
    for i, d in enumerate(reversed(digits)):
        v = int(d)
        if i % 2 == 0:
            v *= 2
        total += v // 10 + v % 10
    return (10 - total % 10) % 10


def cusip_to_isin(cusip: str, country: str = "US") -> str:
    if not is_valid_cusip(cusip):
        raise ValueError(f"invalid CUSIP {cusip!r}")
    body = country + cusip
    return body + str(isin_check_digit(body))


def is_valid_isin(isin: str) -> bool:
    return (len(isin) == 12 and isin[:2].isalpha() and isin[11].isdigit()
            and isin_check_digit(isin[:11]) == int(isin[11]))


def isin_embedded_cusip(isin: str) -> str | None:
    """CUSIP inside an ISIN when the ISIN is CUSIP-based (US, CINS such as AN/BM), else None."""
    if is_valid_isin(isin) and is_valid_cusip(isin[2:11]):
        return isin[2:11]
    return None


# --------------------------------------------------------------------------- HTML parsing
class _RowCollector(HTMLParser):
    """Collect <tr> rows as lists of cell texts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t == "tr":
            self._row = []
        elif t in ("td", "th") and self._row is not None:
            self._cell = []
        elif t == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(_clean("".join(self._cell)))
            self._cell = None
        elif t == "tr" and self._row is not None:
            if self._cell is not None:
                self._row.append(_clean("".join(self._cell)))
                self._cell = None
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


_NUM = re.compile(r"^\d{1,3}(,\d{3})*$")
_FOOTNOTE = re.compile(r"\s*\((?:[a-z](?:,\s*[a-z])*)\)\s*$")


def strip_footnotes(name: str) -> str:
    prev = None
    while prev != name:
        prev = name
        name = _FOOTNOTE.sub("", name).strip()
    return name.rstrip(" *")


def _row_holding(cells: list[str]) -> tuple[str, int, int] | None:
    vals = [c for c in cells if c and c != "$"]
    if len(vals) != 3:
        return None
    name, shares, value = vals
    if not re.search(r"[A-Za-z]", name) or not _NUM.match(shares) or not _NUM.match(value):
        return None
    return name, int(shares.replace(",", "")), int(value.replace(",", ""))


_PAGE_SPLIT = re.compile(r"Schedule\s+of\s+Investments", re.I)
_MONTHS = ("JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER")


def _page_title(chunk_html: str) -> str:
    """Upper-case text right after a 'Schedule of Investments' heading."""
    text = _clean(html.unescape(re.sub(r"<[^>]+>", " ", chunk_html[:6000])))
    return text[:400].upper()


def _fund_pages(doc_html: str, fund_title: str | None):
    """(page number, html) per schedule page. With ``fund_title``, only pages titled exactly
    ``iSHARES® <fund_title> <Month>`` are kept, so 'S&P 500 GROWTH INDEX FUND' is excluded."""
    title_re = None
    if fund_title:
        title_re = re.compile(r"^(?:\W|UNAUDITED|CONTINUED)*ISHARES\W*(?:REG\W*)?"
                              + re.escape(fund_title.upper()) + r"\s+(?:\(CONTINUED\)\s+)?(?:"
                              + _MONTHS + r")\b")
    for page_no, chunk in enumerate(_PAGE_SPLIT.split(doc_html)[1:]):
        if title_re is None or title_re.search(_page_title(chunk)):
            yield page_no, chunk


def parse_schedule(doc_html: str, fund_title: str | None = None) -> list[dict]:
    """Parse the common-stock lines of a schedule of investments.

    Pages start at each 'Schedule of Investments' heading. With ``fund_title`` (multi-fund
    filings), only pages titled exactly ``iSHARES® <fund_title> <Month>`` are kept. Within kept
    pages, rows between a COMMON STOCKS heading and the next TOTAL COMMON STOCKS / SHORT-TERM /
    TOTAL INVESTMENTS heading with exactly (name, shares, value) are holdings; industry headings
    and subtotals (no share count) are skipped.
    """
    holdings: list[dict] = []
    in_stocks = False
    for page_no, chunk in _fund_pages(doc_html, fund_title):
        p = _RowCollector()
        p.feed(chunk)
        for cells in p.rows:
            joined = " ".join(c for c in cells if c).upper()
            if not joined:
                continue
            if joined.startswith("COMMON STOCK"):
                in_stocks = True
                continue
            if joined.startswith(("TOTAL COMMON STOCK", "SHORT-TERM", "SHORT TERM",
                                  "TOTAL INVESTMENTS")):
                in_stocks = False
                continue
            if not in_stocks:
                continue
            # Subtotals carry no share count and are rejected by _row_holding; a name may
            # itself start with "Total" (Total System Services Inc.).
            h = _row_holding(cells)
            if h is None:
                continue
            holdings.append({"raw_name": h[0], "issuer_name": strip_footnotes(h[0]),
                             "shares": h[1], "value_usd": h[2], "page": page_no})
    return holdings


def find_total_common_stocks(doc_html: str, fund_title: str | None = None) -> int | None:
    """Value on the schedule's 'Total Common Stocks' line (used as a parse checksum)."""
    for _, chunk in _fund_pages(doc_html, fund_title):
        p = _RowCollector()
        p.feed(chunk)
        rows = [[c for c in cells if c and c != "$"] for cells in p.rows]
        for k, vals in enumerate(rows):
            if vals and vals[0].upper().startswith("TOTAL COMMON STOCK"):
                # SPY prints the value on the same row; iShares on the following (cost) row.
                for cand in (vals[1:], rows[k + 1] if k + 1 < len(rows) else []):
                    nums = [v for v in cand if _NUM.match(v)]
                    if nums:
                        return int(nums[-1].replace(",", ""))
    return None


# --------------------------------------------------------------------------- 13F list
_13F_LINE = re.compile(r"^([0-9A-Z*@#]{6}) ([0-9A-Z*@#]{2}) (\d)\s+(\*\s+)?(.+?)\s*$")


def parse_13f_text(text: str) -> tuple[dict[str, str], set[str]]:
    """Return (CUSIP -> issuer name, starred CUSIPs).

    Only the issuer-name column is used: pdftotext shifts the description column between
    lines, so it is not relied upon. Issue numbers 90/95 are the list's option series and are
    excluded. The '*' flag accompanies issues with listed options (large equities)."""
    out: dict[str, str] = {}
    starred: set[str] = set()
    for line in text.splitlines():
        m = _13F_LINE.match(line)
        if not m:
            continue
        issuer, issue, chk, star, rest = m.groups()
        if issue in ("90", "95"):
            continue
        cusip = issuer + issue + chk
        if is_valid_cusip(cusip):
            out[cusip] = re.split(r"\s{2,}", rest)[0].strip()
            if star:
                starred.add(cusip)
    return out, starred


# --------------------------------------------------------------------------- name matching
_ABBREV = {
    "INTERNATIONAL": "INTL", "INTERNATL": "INTL", "HOLDINGS": "HLDGS", "HOLDING": "HLDG",
    "CORPORATION": "CORP", "INCORPORATED": "INC", "COMPANY": "CO", "COMPANIES": "COS",
    "MANUFACTURING": "MFG", "SERVICES": "SVCS", "SERVICE": "SVC", "SYSTEMS": "SYS",
    "FINANCIAL": "FINL", "INDUSTRIES": "INDS", "LABORATORIES": "LABS", "AMERICA": "AMER",
    "NATIONAL": "NATL", "MANAGEMENT": "MGMT", "LIMITED": "LTD",
}
_STOP = {"AND", "OF", "THE"}
_SUFFIX = {"INC", "CORP", "CO", "LTD", "PLC", "NV", "SA", "AG", "LP", "LLC", "NEW", "DEL", "DE",
           "COM", "SHS", "ORD", "SE", "NVS", "N", "V"}


def normalize_name(name: str) -> str:
    s = html.unescape(name).upper()
    s = strip_footnotes(s)
    s = re.sub(r"['’`\x92]", "", s)
    s = s.replace("&", " AND ")
    s = re.sub(r"\(THE\)", " ", s)
    s = re.sub(r"\((?:FORMERLY|FKA|F/K/A)\b[^)]*\)?", " ", s)  # 'RS Legacy Corp (formerly Radio'
    s = re.sub(r"/[A-Z]{2,3}/?\s*$", " ", s)  # EDGAR state tags: 'JDS UNIPHASE CORP /CA/'
    s = re.sub(r"\b(?:CL(?:ASS)?|SERIES|SER)\.?\s*[A-C]\b", " ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    toks = [_ABBREV.get(t, t) for t in s.split()]
    toks = [t for t in toks if t not in _STOP]
    # Fixed-width fields truncate legal suffixes ('R S Legacy Corporati'): drop a final
    # token of >= 4 letters that is a prefix of one.
    if len(toks) > 1 and len(toks[-1]) >= 4 and any(
            w.startswith(toks[-1]) for w in ("CORPORATION", "INCORPORATED", "COMPANY", "LIMITED")):
        toks.pop()
    while toks and toks[-1] in _SUFFIX:
        toks.pop()
    return " ".join(toks)


def share_class(name: str) -> str | None:
    m = re.search(r"\b(?:CL(?:ASS)?|SERIES|SER)\.?\s*([A-C])\b", name.upper())
    return m.group(1) if m else None


def _collapse_initials(norm: str) -> str:
    """Join each single-letter token onto the next token (V F -> VF, O REILLY -> OREILLY)."""
    out, buf = [], ""
    for t in norm.split():
        if len(t) == 1:
            buf += t
        else:
            out.append(buf + t)
            buf = ""
    if buf:
        out.append(buf)
    return " ".join(out)


def name_variants(norm: str) -> set[str]:
    """Exact-match keys: the name, its initials-collapsed form and its sorted token bag
    (handles 'WALT DISNEY' vs 'DISNEY WALT', 'E I DU PONT' vs 'DU PONT E I')."""
    if not norm:
        return set()
    joined = re.sub(r"\b([A-Z0-9])\s+(?=[A-Z0-9]\b)", r"\1", norm)  # 'R S LEGACY' -> 'RS LEGACY'
    v = {norm, _collapse_initials(norm), joined}
    v |= {"BAG:" + " ".join(sorted(x.split())) for x in list(v)}
    return v


def _abbrev_of(short: str, full: str) -> bool:
    """``short`` abbreviates ``full``: equal, prefix, or an in-order subsequence sharing the
    first letter (STL ~ STEEL, COMMUNICATNS ~ COMMUNICATIONS)."""
    if short == full or full.startswith(short):
        return True
    if len(short) < 2 or short[0] != full[0]:
        return False
    it = iter(full)
    return all(ch in it for ch in short)


def aligned(query: str, entry: str, entry_raw_len: int) -> bool:
    """Position-by-position abbreviation match. The first tokens must be equal or prefixes of
    each other. The entry may be shorter than the query only when the raw 13F/LSEG name looks
    truncated (>= 22 characters)."""
    q, e = query.split(), entry.split()
    if not q or not e or len(e) > len(q):
        return False
    if len(e) < len(q) and entry_raw_len < 22:
        return False
    if not (q[0] == e[0] or q[0].startswith(e[0]) or e[0].startswith(q[0])):
        return False
    if min(len(q[0]), len(e[0])) < 3 and q[0] != e[0]:
        return False
    return all(_abbrev_of(a, b) or _abbrev_of(b, a) for a, b in zip(e[1:], q[1:]))


def first_tokens_compatible(a: str, b: str) -> bool:
    """Leading name tokens agree: equal, a prefix (>= 3 chars), or an abbreviation (>= 3 chars,
    same first letter, e.g. UTD ~ UNITED, NTL ~ NATL)."""
    if a == b:
        return True
    if min(len(a), len(b)) < 3:
        return False
    return _abbrev_of(a, b) or _abbrev_of(b, a)


@dataclass
class Match:
    key: str | None
    method: str
    detail: str
    candidates: list[str] = field(default_factory=list)


class NameIndex:
    """Name -> key lookup (key = CUSIP issuer number for the 13F list, ISIN for LSEG names)."""

    def __init__(self, entries: list[tuple[str, str]], preferred: set[str] | None = None):
        self.names: dict[str, list[str]] = defaultdict(list)
        self.variant_to_keys: dict[str, set[str]] = defaultdict(set)
        self.norm_entries: list[tuple[str, str, int]] = []
        self.preferred = preferred or set()
        for key, raw in entries:
            if raw not in self.names[key]:
                self.names[key].append(raw)
            n = normalize_name(raw)
            for v in name_variants(n):
                self.variant_to_keys[v].add(key)
            if n:
                self.norm_entries.append((key, n, len(raw)))

    def _decide(self, keys: set[str], method: str) -> Match:
        if len(keys) == 1:
            k = next(iter(keys))
            return Match(k, method, f"'{self.names[k][0]}'")
        pref = keys & self.preferred
        if len(pref) == 1:
            k = next(iter(pref))
            return Match(k, method + "+optioned_tiebreak",
                         f"'{self.names[k][0]}'; only optioned issuer among {sorted(keys)}")
        return Match(None, "ambiguous", f"{method}: several candidates", sorted(keys)[:25])

    def match(self, name: str) -> Match:
        norm = normalize_name(name)
        if not norm:
            return Match(None, "unmatched", "empty normalized name")
        keys = set()
        for v in name_variants(norm):
            keys |= self.variant_to_keys.get(v, set())
        if keys:
            return self._decide(keys, "exact_name")
        qs = {norm, _collapse_initials(norm)}
        keys = {k for k, n, ln in self.norm_entries
                if any(aligned(q, n, ln) or aligned(n, q, len(name)) for q in qs)}
        if keys:
            return self._decide(keys, "aligned_name")
        return Match(None, "unmatched", "no compatible name")


def build_13f_index(cusip_names: dict[str, str], starred: set[str]) -> NameIndex:
    per_issuer: dict[str, list[str]] = defaultdict(list)
    for cusip, nm in sorted(cusip_names.items()):
        if nm not in per_issuer[cusip[:6]]:
            per_issuer[cusip[:6]].append(nm)
    entries = [(iss, nm) for iss, nms in per_issuer.items() for nm in nms]
    return NameIndex(entries, preferred={c[:6] for c in starred})


def equity_cusip(issuer6: str, cusip_names: dict[str, str], starred: set[str],
                 cls: str | None) -> tuple[str | None, str]:
    """The single equity CUSIP of an issuer (numeric issue numbers only; letters are debt).
    With several numeric issues, an optioned ('*') one is taken only if it is unique and the
    holding names no share class."""
    issues = sorted(c for c in cusip_names if c[:6] == issuer6 and c[6:8].isdigit())
    if len(issues) == 1:
        return issues[0], "single equity issue"
    star = [c for c in issues if c in starred]
    if len(star) == 1 and cls is None:
        return star[0], f"only optioned of {issues}"
    return None, f"equity issues {issues}; class {cls or '-'} not resolvable from the list"


# --------------------------------------------------------------------------- downloads
def _ua() -> str:
    return os.environ.get("SEC_USER_AGENT", DEFAULT_UA)


def fetch(url: str, dest: Path, offline: bool) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if offline:
        raise FileNotFoundError(f"{dest} not cached and --offline given")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": _ua(), "Accept-Encoding": "identity"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:  # EDGAR FTS returns sporadic 500s; 429 = throttle
            if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    if url.endswith(".pdf") and not data.startswith(b"%PDF"):
        raise RuntimeError(f"{url} did not return a PDF")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    time.sleep(0.25)  # SEC fair access: far below 10 requests/second
    return dest


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_html(p: Path) -> str:
    raw = p.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def load_13f(offline: bool) -> tuple[dict[str, str], set[str], dict]:
    pdf = fetch(THIRTEENF_URL, SEC_CACHE / "13f_list" / "13flist2010q4.pdf", offline)
    txt = pdf.with_suffix(".txt")
    if not txt.exists():
        exe = shutil.which("pdftotext")
        if not exe:
            raise RuntimeError("pdftotext (xpdf/poppler) is required to read the 13F list PDF")
        subprocess.run([exe, "-layout", str(pdf), str(txt)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    names, starred = parse_13f_text(txt.read_text(encoding="latin-1"))
    meta = {"url": THIRTEENF_URL, "sha256_pdf": sha256(pdf), "list_current_as_of": "2010-12-15",
            "cusips_parsed": len(names),
            "note": "CUSIP data (c) American Bankers Association; cached locally, not redistributed."}
    return names, starred, meta


def spy_filing_inventory(offline: bool) -> dict:
    p = fetch(SUBMISSIONS_URL, SEC_CACHE / "submissions" / "CIK0000884394.json", offline)
    d = json.loads(p.read_text(encoding="utf-8"))
    r = d["filings"]["recent"]
    rows = [{"form": f, "filing_date": fd, "report_date": rd, "accession": a}
            for f, fd, rd, a in zip(r["form"], r["filingDate"], r["reportDate"], r["accessionNumber"])
            if "2010-07-01" <= fd <= "2011-12-31"]
    return {"url": SUBMISSIONS_URL, "sha256": sha256(p),
            "n_q_filings_ever": sum(f.startswith("N-Q") for f in r["form"]),
            "filings_2010H2_2011": rows}


# --------------------------------------------------------------------------- LSEG side
def load_identifiers() -> dict[str, list[str]]:
    rics: dict[str, list[str]] = defaultdict(list)
    for f in sorted(LSEG_DIR.glob("identifiers_*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("ISIN"):
                    rics[row["ISIN"]].append(row["RIC"])
    return rics


def load_lseg_names(rics: dict[str, list[str]]) -> dict[str, list[str]]:
    """ISIN -> names from LSEG snapshot files (constituent names by RIC, common names by ISIN)."""
    ric_isin = {r: i for i, rs in rics.items() for r in rs}
    names: dict[str, list[str]] = defaultdict(list)

    def add(isin, nm):
        if isin and nm and nm not in names[isin]:
            names[isin].append(nm)

    for f in sorted(LSEG_DIR.glob("members_*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                add(ric_isin.get(row.get("Constituent RIC") or ""), row.get("Constituent Name"))
    for pat in ("**/reference_snapshot.csv", "**/sp500_members_at_start.csv"):
        for f in sorted(LSEG_ROOT.glob(pat)):
            with f.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    add(row.get("ISIN"), row.get("Company Common Name"))
    return names


def load_replay_sets(sessions: list[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {s: set() for s in sessions}
    with gzip.open(LSEG_DIR / "membership_dated_candidate.csv.gz", "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["session"] in out:
                out[row["session"]].add(row["isin"])
    return out


def load_prices(rics_wanted: set[str], sessions: list[str]) -> dict[tuple[str, str], float]:
    """(RIC, session) -> unadjusted LSEG close (TRDPRC_1) from the equities_*.csv files."""
    out: dict[tuple[str, str], float] = {}
    want = set(sessions)
    for f in sorted(LSEG_DIR.glob("equities_*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            ric_row, field_row = next(rd), next(rd)
            cols = [(j, ric_row[j]) for j in range(1, len(ric_row))
                    if field_row[j] == "TRDPRC_1" and ric_row[j] in rics_wanted]
            if not cols:
                continue
            for row in rd:
                if row and row[0] in want:
                    for j, ric in cols:
                        if j < len(row) and row[j]:
                            out[(ric, row[0])] = float(row[j])
    return out


def load_changes() -> list[dict]:
    with (LSEG_DIR / "membership_changes.csv").open(newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["Date"] <= "2011-06-30"]


# --------------------------------------------------------------------------- linking
def resolve_holdings(holdings: list[dict], src: Source, idx13: NameIndex,
                     cusip_names: dict[str, str], starred: set[str]) -> None:
    for h in holdings:
        m = idx13.match(h["issuer_name"])
        cusip, why = (equity_cusip(m.key, cusip_names, starred, share_class(h["issuer_name"]))
                      if m.key else (None, ""))
        h.update({
            "fund": src.fund, "as_of_date": src.as_of, "source_form": src.form,
            "source_accession": src.accession, "source_url": src.url,
            "issuer_cusip6": m.key, "cusip": cusip,
            "isin_2010": cusip_to_isin(cusip) if cusip and cusip[0].isdigit() else None,
            "name_13f": idx13.names[m.key][0] if m.key else None,
            "match_13f": m.method,
            "match_13f_detail": "; ".join(x for x in (m.detail, why,
                                          f"candidates {m.candidates}" if m.candidates else "") if x),
            "implied_price": round(h["value_usd"] / h["shares"], 4) if h["shares"] else None,
        })


def _ratio(h: dict, isin: str, rics: dict[str, list[str]], prices: dict, session: str | None):
    """Best (RIC, close, log-ratio) among an ISIN's RICs, closest to the implied price."""
    if not session or not h.get("implied_price"):
        return None
    best = None
    for r in rics.get(isin, []):
        p = prices.get((r, session))
        if p:
            lr = math.log(p / h["implied_price"])
            if best is None or abs(lr) < abs(best[2]):
                best = (r, p, lr)
    return best


def link(holdings: list[dict], replay: set[str], rics: dict[str, list[str]],
         lseg_names: dict[str, list[str]], prices: dict, session: str | None) -> dict:
    """One-to-one holding-line <-> replay-ISIN links. Mutates holdings with link fields."""
    by_iss: dict[str, list[str]] = defaultdict(list)
    for i in sorted(replay):
        c = isin_embedded_cusip(i)
        if c:
            by_iss[c[:6]].append(i)
    lines_per_iss: dict[str, list[int]] = defaultdict(list)
    for n, h in enumerate(holdings):
        h.update({"replay_isin": None, "link_method": None, "lseg_ric": None,
                  "lseg_close": None, "price_log_ratio": None})
        if h["issuer_cusip6"]:
            lines_per_iss[h["issuer_cusip6"]].append(n)

    used: set[str] = set()
    multi_line_issuers = {}

    def set_link(h, isin, method):
        h["replay_isin"], h["link_method"] = isin, method
        used.add(isin)
        r = _ratio(h, isin, rics, prices, session)
        if r:
            h["lseg_ric"], h["lseg_close"], h["price_log_ratio"] = r[0], r[1], round(r[2], 4)

    # 1. CUSIP issuer number
    for iss, idxs in lines_per_iss.items():
        cands = by_iss.get(iss, [])
        if not cands:
            continue
        if len(idxs) == 1 and len(cands) == 1:
            set_link(holdings[idxs[0]], cands[0], "cusip_issuer")
            continue
        # multi-line issuer (share classes) or multi-ISIN issuer: pair by exact 2010 ISIN, then
        # by price; leave the rest for review.
        multi_line_issuers[iss] = {"lines": [holdings[i]["issuer_name"] for i in idxs],
                                   "replay_isins": cands}
        free = list(cands)
        for i in idxs:
            h = holdings[i]
            if h["isin_2010"] in free:
                set_link(h, h["isin_2010"], "cusip_exact_isin")
                free.remove(h["isin_2010"])
        for i in idxs:
            h = holdings[i]
            if h["replay_isin"] or not free:
                continue
            scored = [(abs(r[2]), isin) for isin in free
                      if (r := _ratio(h, isin, rics, prices, session)) and abs(r[2]) <= PRICE_TOL]
            if len(scored) == 1 or (len(scored) > 1 and sorted(scored)[1][0] > 3 * sorted(scored)[0][0]):
                isin = sorted(scored)[0][1]
                set_link(h, isin, "cusip_issuer+price_class")
                free.remove(isin)

    # 2. LSEG names, one-to-one among the still-unlinked
    rest_isins = sorted(i for i in replay if i not in used)
    lidx = NameIndex([(i, nm) for i in rest_isins for nm in lseg_names.get(i, [])])
    proposals: dict[str, list[int]] = defaultdict(list)
    for n, h in enumerate(holdings):
        if h["replay_isin"]:
            continue
        m = lidx.match(h["issuer_name"])
        if m.key:
            proposals[m.key].append(n)
    for isin, ns in proposals.items():
        if len(ns) == 1:
            h = holdings[ns[0]]
            r = _ratio(h, isin, rics, prices, session)
            ok = r is not None and abs(r[2]) <= PRICE_TOL
            set_link(h, isin, "lseg_name+price" if ok else
                     ("lseg_name_price_conflict" if r else "lseg_name_no_price"))

    # 2b. first name token compatible with an LSEG name AND price within 2%, one-to-one.
    if session:
        rest_isins = [i for i in sorted(replay) if i not in used]
        pairs = []
        for h in holdings:
            if h["replay_isin"]:
                continue
            q = normalize_name(h["issuer_name"]).split()
            for i in rest_isins:
                firsts = {normalize_name(nm).split()[0] for nm in lseg_names.get(i, [])
                          if normalize_name(nm)}
                if not q or not any(first_tokens_compatible(q[0], f) for f in firsts):
                    continue
                r = _ratio(h, i, rics, prices, session)
                if r and abs(r[2]) <= 0.02:
                    pairs.append((h, i))
        hc, ic = defaultdict(int), defaultdict(int)
        for h, i in pairs:
            hc[id(h)] += 1
            ic[i] += 1
        for h, i in pairs:
            if hc[id(h)] == 1 and ic[i] == 1:
                set_link(h, i, "lseg_first_token+price")

    # 3. price-only candidates (review list, not links)
    rest_isins = [i for i in replay if i not in used]
    price_cands = []
    if session:
        hold_rest = [h for h in holdings if not h["replay_isin"]]
        pairs = []
        for h in hold_rest:
            for i in rest_isins:
                r = _ratio(h, i, rics, prices, session)
                if r and abs(r[2]) <= 0.02:
                    pairs.append((h, i, r))
        hc = defaultdict(int)
        ic = defaultdict(int)
        for h, i, _ in pairs:
            hc[id(h)] += 1
            ic[i] += 1
        for h, i, r in pairs:
            price_cands.append({"holding": h["issuer_name"], "implied_price": h["implied_price"],
                                "replay_isin": i, "rics": rics.get(i, []),
                                "lseg_names": lseg_names.get(i, []), "lseg_close": r[1],
                                "price_log_ratio": round(r[2], 4),
                                "one_to_one": hc[id(h)] == 1 and ic[i] == 1})
    return {"multi_line_issuers": multi_line_issuers, "price_candidates": price_cands}


STRONG_METHODS = ("cusip_issuer", "cusip_exact_isin", "cusip_issuer+price_class", "lseg_name+price")


def apply_confirmations(holdings: list[dict], conf: dict) -> None:
    """Record SEC-confirmed pairs on the holding lines (sec_* columns). Review-tier links that
    SEC confirms are kept; links that SEC contradicts are replaced and flagged."""
    by_name = {r["issuer_name"]: r for r in conf["confirmed"]}
    for h in holdings:
        h.setdefault("sec_confirmation", None)
        h.setdefault("sec_evidence", None)
        r = by_name.get(h["issuer_name"])
        if not r or (h["link_method"] in STRONG_METHODS):
            continue
        ev = r["evidence"]
        h["sec_confirmation"] = r["method"]
        h["sec_evidence"] = (f"CIK {ev['cik']}" if r["method"] == "same_registrant" else
                             f"{ev['form']} {ev['accession']} ({ev['filing_date']}) by CIK "
                             f"{ev['successor_cik']} names '{ev['predecessor_name_found']}'")
        if h["replay_isin"] and h["replay_isin"] != r["replay_isin"]:
            h["link_method"] = f"sec_{r['method']} (replaced {h['replay_isin']})"
        else:
            h["link_method"] = f"sec_{r['method']}"
        h["replay_isin"] = r["replay_isin"]


def after_counts(holdings: list[dict], replay: set[str]) -> dict:
    linked = [h for h in holdings if h["replay_isin"]]
    strong = [h for h in linked if h["link_method"] in STRONG_METHODS]
    sec = [h for h in linked if str(h["link_method"]).startswith("sec_")]
    used = {h["replay_isin"] for h in linked}
    return {"linked_total": len(linked), "linked_strong_cusip_or_name_price": len(strong),
            "linked_sec_confirmed": len(sec),
            "linked_unconfirmed_review": len(linked) - len(strong) - len(sec),
            "holding_lines_unlinked": len(holdings) - len(linked),
            "replay_isins_unlinked": len(replay - used),
            "duplicate_replay_isins": len(linked) - len(used)}


def summarize(src: Source, holdings: list[dict], replay: set[str], rics, lseg_names,
              idx13: NameIndex, linkinfo: dict, presence: dict[str, dict[str, bool]],
              session: str, price_session: str | None) -> dict:
    linked = [h for h in holdings if h["replay_isin"]]
    strong = [h for h in linked if h["link_method"] in
              ("cusip_issuer", "cusip_exact_isin", "cusip_issuer+price_class", "lseg_name+price")]
    weak = [h for h in linked if h not in strong]
    used = {h["replay_isin"] for h in linked}

    def hold_ev(h):
        return {k: h.get(k) for k in ("issuer_name", "shares", "value_usd", "implied_price",
                                      "cusip", "isin_2010", "name_13f", "match_13f",
                                      "match_13f_detail", "replay_isin", "link_method",
                                      "lseg_ric", "lseg_close", "price_log_ratio")}

    def replay_ev(isin):
        c = isin_embedded_cusip(isin)
        iss = c[:6] if c else None
        return {"isin": isin, "rics": rics.get(isin, []), "lseg_names": lseg_names.get(isin, []),
                "embedded_cusip": c, "name_13f_2010q4": idx13.names[iss][0] if iss in idx13.names else None,
                "held_by_issuer_number": presence.get(iss, {}) if iss else {}}

    def held_elsewhere(h):
        return presence.get(h["issuer_cusip6"], {}) if h["issuer_cusip6"] else {}

    price_conflicts = [hold_ev(h) for h in strong
                       if h["price_log_ratio"] is not None and abs(h["price_log_ratio"]) > PRICE_TOL]
    no_price = [hold_ev(h) for h in strong if h["price_log_ratio"] is None] if price_session else []
    only_hold = [{**hold_ev(h), "issuer_held_in": held_elsewhere(h)}
                 for h in holdings if not h["replay_isin"]]
    only_replay = [replay_ev(i) for i in sorted(replay - used)]
    return {
        "source": src.key, "fund": src.fund, "holdings_as_of": src.as_of,
        "replay_session": session, "lseg_price_session": price_session,
        "counts": {
            "holding_lines": len(holdings),
            "holding_lines_13f_resolved": sum(1 for h in holdings if h["issuer_cusip6"]),
            "replay_isins": len(replay),
            "linked_total": len(linked),
            "linked_strong": len(strong),
            "linked_same_isin_as_2010_cusip": sum(1 for h in linked if h["isin_2010"] == h["replay_isin"]),
            "linked_by_method": {m: sum(1 for h in linked if h["link_method"] == m)
                                 for m in sorted({h["link_method"] for h in linked})},
            "linked_needing_review": len(weak),
            "strong_links_with_price_conflict": len(price_conflicts),
            "strong_links_without_price": len(no_price),
            "holding_lines_unlinked": len(only_hold),
            "replay_isins_unlinked": len(only_replay),
            "price_only_candidates_one_to_one": sum(1 for c in linkinfo["price_candidates"] if c["one_to_one"]),
        },
        "price_log_ratio_abs_max_strong": max((abs(h["price_log_ratio"]) for h in strong
                                               if h["price_log_ratio"] is not None), default=None),
        "links_needing_review": [hold_ev(h) for h in weak],
        "strong_links_with_price_conflict": price_conflicts,
        "strong_links_without_price": no_price,
        "only_in_holdings": only_hold,
        "only_in_replay": only_replay,
        "price_only_candidates": linkinfo["price_candidates"],
        "multi_line_issuers": linkinfo["multi_line_issuers"],
    }


# --------------------------------------------------------------------------- SEC confirmation
# Unresolved / review-tier pairs are confirmed from SEC records only:
#   CUSIP -> issuer CIK: EDGAR full-text search of Schedule 13G/13D cover pages, where the
#            subject company is the one entity present in (nearly) every hit;
#   ticker -> CIK:       SEC company_tickers.json (active listings only), as a fallback;
#   CIK names:           data.sec.gov submissions JSON (current name + formerNames);
#   succession:          the successor's own 8-K12B / 8-K12G3 filing naming the predecessor.
FTS_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{q}%22&forms={forms}"
FTS_13G_FORMS = "SC%2013G,SC%2013G/A,SC%2013D,SC%2013D/A,SCHEDULE%2013G,SCHEDULE%2013G/A"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
CIK_LOOKUP_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
FTS_ISSUER_FORMS = "8-K,8-K12B,8-K12G3,8-A12B,10-K,10-Q,S-4,424B3,DEFM14A,DEF%2014A"
DELIST_FORMS = ("25-NSE", "25", "15-12B", "15-12G", "15-15D")
SUCCESSION_FORMS = ("8-K12B", "8-K12G3", "8-K12B/A", "8-K12G3/A")


def fetch_json(url: str, dest: Path, offline: bool):
    p = fetch(url, dest, offline)
    return json.loads(p.read_text(encoding="utf-8"))


def subject_from_hits(hits: list[dict]) -> tuple[str | None, str, int, list[str]]:
    """The subject company of 13G/13D hits: the CIK present in at least 60% of hits and at
    least twice as often as any other CIK (filers vary; a parent's subsidiary or plan may
    co-appear). Returns (cik, display name, hit count, sample accessions)."""
    if not hits:
        return None, "", 0, []
    cnt: dict[str, int] = defaultdict(int)
    names: dict[str, str] = {}
    for h in hits:
        src = h["_source"]
        for c, n in zip(src.get("ciks", []), src.get("display_names", [])):
            cnt[c] += 1
            names[c] = n
    best = sorted(cnt.items(), key=lambda kv: -kv[1])
    top_cik, top_n = best[0]
    if top_n < 0.6 * len(hits) or (len(best) > 1 and best[1][1] * 2 > top_n):
        return None, "", len(hits), []
    acc = sorted({h["_id"].split(":")[0] for h in hits
                  if top_cik in h["_source"].get("ciks", [])})[:3]
    return top_cik.lstrip("0"), names[top_cik], len(hits), acc


class SecResolver:
    def __init__(self, offline: bool):
        self.offline = offline
        self._subs: dict[str, dict] = {}
        self._tickers: dict[str, str] | None = None

    def cusip_cik(self, cusip: str) -> dict:
        try:
            d = fetch_json(FTS_URL.format(q=cusip, forms=FTS_13G_FORMS),
                           SEC_CACHE / "fts" / f"13g_{cusip}.json", self.offline)
        except (urllib.error.HTTPError, FileNotFoundError) as e:
            return {"query": cusip, "cik": None, "error": str(e)}
        hits = d.get("hits", {}).get("hits", [])
        cik, name, n, acc = subject_from_hits(hits)
        return {"query": cusip, "cik": cik, "display_name": name, "hits": n,
                "sample_accessions": acc, "source": "EDGAR FTS Schedule 13G/13D"}

    def ticker_cik(self, ticker: str) -> str | None:
        if self._tickers is None:
            d = fetch_json(TICKERS_URL, SEC_CACHE / "company_tickers.json", self.offline)
            self._tickers = {v["ticker"].upper(): str(v["cik_str"]) for v in d.values()}
        return self._tickers.get(ticker.upper())

    def submissions(self, cik: str) -> dict:
        if cik in self._subs:
            return self._subs[cik]
        c10 = cik.zfill(10)
        d = fetch_json(f"https://data.sec.gov/submissions/CIK{c10}.json",
                       SEC_CACHE / "submissions" / f"CIK{c10}.json", self.offline)
        filings = [dict(zip(d["filings"]["recent"].keys(), row))
                   for row in zip(*d["filings"]["recent"].values())]
        for f in d["filings"].get("files", []):
            extra = fetch_json(f"https://data.sec.gov/submissions/{f['name']}",
                               SEC_CACHE / "submissions" / f["name"], self.offline)
            filings += [dict(zip(extra.keys(), row)) for row in zip(*extra.values())]
        out = {"cik": cik, "name": d.get("name"), "tickers": d.get("tickers", []),
               "former_names": d.get("formerNames", []), "filings": filings}
        self._subs[cik] = out
        return out

    def name_ciks(self, name: str) -> dict:
        """CIK for an entity name via SEC cik-lookup-data.txt (current and former names), exact
        normalized match, kept only if exactly one candidate has filed 10-K/10-Q/20-F reports."""
        if not hasattr(self, "_lookup"):
            p = fetch(CIK_LOOKUP_URL, SEC_CACHE / "cik-lookup-data.txt", self.offline)
            self._lookup: dict[str, set[str]] = defaultdict(set)
            for line in p.read_text(encoding="latin-1").splitlines():
                m = re.match(r"^(.*):(\d{10}):\s*$", line)
                if m:
                    self._lookup[normalize_name(m.group(1))].add(m.group(2).lstrip("0"))
        cands = sorted(self._lookup.get(normalize_name(name), set()))
        filers = [c for c in cands[:8] if any(
            f.get("form") in ("10-K", "10-Q", "20-F", "10-K405") for f in self.submissions(c)["filings"])]
        return {"query": name, "cik": filers[0] if len(filers) == 1 else None,
                "candidates": cands[:8], "periodic_filers": filers,
                "source": "SEC cik-lookup-data.txt"}

    def id_cited_by(self, ident: str) -> list[dict]:
        """Issuer-type filings (8-K, 10-K, ...) whose text contains ``ident`` (CUSIP or ISIN)."""
        try:
            d = fetch_json(FTS_URL.format(q=ident, forms=FTS_ISSUER_FORMS),
                           SEC_CACHE / "fts" / f"issuer_{ident}.json", self.offline)
        except (urllib.error.HTTPError, FileNotFoundError):
            return []
        return [{"ciks": [c.lstrip("0") for c in h["_source"].get("ciks", [])],
                 "form": h["_source"].get("form"), "file_date": h["_source"].get("file_date"),
                 "accession": h["_id"].split(":")[0]}
                for h in d.get("hits", {}).get("hits", [])]

    def fund_reported_names(self, ident: str, max_docs: int = 4) -> list[dict]:
        """Issuer names that funds report next to ``ident`` in XML holdings tables
        (13F-HR <nameOfIssuer>, N-PX <issuerName>). Weaker evidence than issuer filings."""
        try:
            # No form filter: EDGAR FTS drops some 13F-HR hits when one is given. XML
            # holdings tables are selected by file name below.
            d = fetch_json(f"https://efts.sec.gov/LATEST/search-index?q=%22{ident}%22",
                           SEC_CACHE / "fts" / f"any_{ident}.json", self.offline)
        except (urllib.error.HTTPError, FileNotFoundError):
            return []
        out = []
        for h in d.get("hits", {}).get("hits", []):
            acc, _, fname = h["_id"].partition(":")
            if not fname.lower().endswith(".xml") or not h["_source"].get("ciks"):
                continue
            cik = h["_source"]["ciks"][0].lstrip("0")
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{fname}"
            try:
                text = read_html(fetch(url, SEC_CACHE / acc / fname, self.offline))
            except (urllib.error.HTTPError, FileNotFoundError):
                continue
            name = record_issuer_name(text, ident)
            if name:
                out.append({"name": name, "form": h["_source"].get("form"),
                            "accession": acc, "url": url})
            if len(out) >= max_docs:
                break
        return out

    def all_names(self, cik: str) -> list[str]:
        s = self.submissions(cik)
        return [s["name"]] + [f["name"] for f in s["former_names"]]

    def succession_filings(self, cik: str) -> list[dict]:
        return [f for f in self.submissions(cik)["filings"] if f.get("form") in SUCCESSION_FORMS]

    def filing_text(self, cik: str, f: dict) -> str:
        acc = f["accessionNumber"]
        url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/"
               f"{f['primaryDocument']}")
        p = fetch(url, SEC_CACHE / acc / f["primaryDocument"], self.offline)
        return read_html(p)


_RECORD_OPEN = re.compile(r"<(?:\w+:)?(infoTable|proxyTable)\b[^>]*>", re.I)


def record_issuer_name(xml: str, ident: str) -> str | None:
    """Issuer name from the *same* XML holdings record (13F <infoTable>, N-PX <proxyTable>)
    whose identifier field equals ``ident``; None if no such record."""
    starts = [m for m in _RECORD_OPEN.finditer(xml)]
    for k, m in enumerate(starts):
        end = starts[k + 1].start() if k + 1 < len(starts) else len(xml)
        rec = xml[m.start():end]
        ids = re.findall(r"<(?:\w+:)?(?:cusip|isin)>\s*([^<]+?)\s*<", rec, re.I)
        if any(i.strip().upper() == ident.upper() for i in ids):
            nm = re.search(r"<(?:\w+:)?(?:nameOfIssuer|issuerName)>\s*([^<]+?)\s*<", rec, re.I)
            if nm:
                return html.unescape(nm.group(1))
    return None


def _date(s: str):
    import datetime
    return datetime.date.fromisoformat(s[:10])


def distinctive_name(name: str) -> bool:
    """A normalized name usable as text evidence: >= 2 words or a single word of >= 6
    characters, so short common words ('NEWS' from 'NEWS CORP', 'ITT') do not count."""
    n = normalize_name(name)
    return len(n.split()) >= 2 or len(n) >= 6


def _norm_text(s: str) -> str:
    s = html.unescape(re.sub(r"<[^>]+>", " ", s)).upper().replace("&", " AND ")
    return " " + " ".join(re.sub(r"[^A-Z0-9 ]", " ", s).split()) + " "


def names_compatible(name: str, candidates: list[str]) -> bool:
    """True if ``name`` matches any of one entity's names (all indexed under a single key)."""
    return NameIndex([("entity", c) for c in candidates]).match(name).key is not None


def confirm_pairs(holdings: list[dict], replay_unlinked: list[str], rics: dict[str, list[str]],
                  cusip_names: dict[str, str], res: SecResolver,
                  lseg_names: dict[str, list[str]] | None = None) -> dict:
    """Pair unresolved holding lines with unresolved replay ISINs using SEC records only.

    ``holdings`` are the unlinked lines plus the review-tier links (which are re-checked).
    A pair is confirmed when the 2010 issuer's CIK equals the replay ISIN's CIK
    (``same_registrant``), or the replay CIK filed an 8-K12B/8-K12G3 naming the 2010 issuer
    under any of its SEC names (``succession_filing``)."""
    def h_cik(h):
        cands = ([h["cusip"]] if h.get("cusip") else
                 sorted(c for c in cusip_names if h.get("issuer_cusip6") and c[:6] == h["issuer_cusip6"]
                        and c[6:8].isdigit()))
        lookups = [res.cusip_cik(c) for c in cands]
        for nm in (h["issuer_name"], h.get("name_13f")):
            if not any(r["cik"] for r in lookups) and nm:
                lookups.append(res.name_ciks(nm))
        for r in lookups:
            if r["cik"]:
                names = res.all_names(r["cik"])
                r["sec_names"] = names
                r["name_check"] = names_compatible(h["issuer_name"], names)
                return r
        return {"query": cands, "cik": None, "attempts": lookups}

    lseg_names = lseg_names or {}

    def r_cik(isin):
        c = isin_embedded_cusip(isin)
        if c:
            r = res.cusip_cik(c)
            if r["cik"]:
                return r
        for ric in rics.get(isin, []):
            root = ric.split(".")[0]
            if "^" not in ric:
                cik = res.ticker_cik(root.replace("_", "-"))
                if cik:
                    return {"query": ric, "cik": cik, "source": "SEC company_tickers.json"}
        cands = []
        for nm in lseg_names.get(isin, []):
            r = res.name_ciks(nm)
            if r["cik"]:
                return r
            cands += [x for x in r["periodic_filers"] if x not in cands]
        # Several CIKs share the name (e.g. 'ALLERGAN PLC'): each is tested, and only a
        # unique SEC-evidenced pairing survives the one-to-one check.
        return {"query": c or isin, "cik": None, "cik_candidates": cands}

    hside = [(h, h_cik(h)) for h in holdings]
    rside = {i: r_cik(i) for i in replay_unlinked}
    for h in holdings:  # review-tier links: re-check their own replay ISIN as well
        if h.get("replay_isin") and h["replay_isin"] not in rside:
            rside[h["replay_isin"]] = r_cik(h["replay_isin"])

    def test_pair(hc: dict, replay_cik: str):
        """(method, evidence) if SEC records tie the 2010 CIK to the replay CIK, else None."""
        if replay_cik == hc["cik"]:
            return "same_registrant", {"cik": hc["cik"]}
        h_names = res.all_names(hc["cik"])
        for f in res.succession_filings(replay_cik):
            text = _norm_text(res.filing_text(replay_cik, f))
            hit = next((n for n in h_names if distinctive_name(n)
                        and f" {normalize_name(n)} " in text), None)
            if hit:
                return "succession_filing", {
                    "successor_cik": replay_cik, "predecessor_cik": hc["cik"], "form": f["form"],
                    "accession": f["accessionNumber"], "filing_date": f["filingDate"],
                    "predecessor_name_found": hit}
        return None

    proposals = []
    for h, hc in hside:
        if not hc.get("cik") or not hc.get("name_check"):
            continue
        for isin, rc in rside.items():
            for cik in ([rc["cik"]] if rc.get("cik") else rc.get("cik_candidates", [])):
                t = test_pair(hc, cik)
                if t:
                    proposals.append((h, isin, t[0], {**t[1], "replay_cik": cik}))
                    break

    # Second pass, open pairs only: the replay CIK's own 8-K reporting completion of an
    # acquisition or merger (Item 2.01) that names the 2010 issuer.
    proposed_h = {id(h) for h, *_ in proposals}
    proposed_i = {i for _, i, *_ in proposals}
    for isin, rc in rside.items():
        if isin in proposed_i:
            continue
        for cik in ([rc["cik"]] if rc.get("cik") else rc.get("cik_candidates", [])):
            mergers = [f for f in res.submissions(cik)["filings"]
                       if f.get("form") == "8-K" and "2.01" in str(f.get("items", ""))]
            for h, hc in hside:
                if id(h) in proposed_h or not hc.get("cik") or not hc.get("name_check"):
                    continue
                names = [n for n in res.all_names(hc["cik"]) if distinctive_name(n)]
                delist = [d["filingDate"] for d in res.submissions(hc["cik"])["filings"]
                          if d.get("form") in DELIST_FORMS]
                for f in mergers:
                    if not any(abs((_date(d) - _date(f["filingDate"])).days) <= 45 for d in delist):
                        continue  # predecessor did not delist around this 8-K
                    text = _norm_text(res.filing_text(cik, f))
                    hit = next((n for n in names if f" {normalize_name(n)} " in text), None)
                    if hit:
                        proposals.append((h, isin, "merger_completion_8k",
                                          {"successor_cik": cik, "predecessor_cik": hc["cik"],
                                           "form": "8-K item 2.01", "accession": f["accessionNumber"],
                                           "filing_date": f["filingDate"],
                                           "predecessor_name_found": hit,
                                           "predecessor_delisting_dates": delist}))
                        proposed_h.add(id(h))
                        break
    # Fallback: the 2010 issuer's own filing cites the replay CUSIP/ISIN (e.g. a redomicile or
    # rename 8-K giving the new identifier). Only for pairs not proposed above.
    proposed_h = {id(h) for h, *_ in proposals}
    proposed_i = {i for _, i, *_ in proposals}
    open_h = {hc["cik"]: h for h, hc in hside
              if hc.get("cik") and hc.get("name_check") and id(h) not in proposed_h}
    for isin in rside:
        if isin in proposed_i:
            continue
        ident = isin_embedded_cusip(isin) or isin
        for hit in res.id_cited_by(ident):
            match = [c for c in hit["ciks"] if c in open_h]
            if match:
                proposals.append((open_h[match[0]], isin, "issuer_filing_cites_replay_id",
                                  {"cik": match[0], "identifier": ident, "form": hit["form"],
                                   "accession": hit["accession"], "filing_date": hit["file_date"]}))
                break

    # Fallback (weakest tier): names that funds report beside the replay identifier in XML
    # holdings tables. A reported name -> candidate CIKs is tested with test_pair; failing that,
    # a reported name equal to one of the open 2010 CIK's SEC names is accepted as
    # 'fund_reported_name' (clearly labelled).
    proposed_h = {id(h) for h, *_ in proposals}
    proposed_i = {i for _, i, *_ in proposals}
    open_hc = [(h, hc) for h, hc in hside
               if hc.get("cik") and hc.get("name_check") and id(h) not in proposed_h]
    for isin in sorted(rside):
        if isin in proposed_i or not open_hc:
            continue
        ident = isin_embedded_cusip(isin) or isin
        reported = res.fund_reported_names(ident)
        rside[isin]["fund_reported_names"] = reported
        names = sorted({r["name"] for r in reported})
        found = None
        for nm in names:
            for cand in res.name_ciks(nm)["periodic_filers"]:
                for h, hc in open_hc:
                    t = test_pair(hc, cand)
                    if t:
                        found = (h, isin, t[0], {**t[1], "via_fund_reported_name": nm,
                                                 "fund_documents": [r["url"] for r in reported]})
                        break
                if found:
                    break
            if found:
                break
        if not found:
            hits = [(h, hc) for h, hc in open_hc
                    if any(names_compatible(nm, res.all_names(hc["cik"])) for nm in names)]
            if len(hits) == 1:
                h, hc = hits[0]
                found = (h, isin, "fund_reported_name",
                         {"cik": hc["cik"], "identifier": ident, "reported_names": names,
                          "fund_documents": [r["url"] for r in reported]})
        if found:
            proposals.append(found)

    hc_n, ic_n = defaultdict(int), defaultdict(int)
    for h, i, _, _ in proposals:
        hc_n[id(h)] += 1
        ic_n[i] += 1
    confirmed, conflicts = [], []
    for h, i, how, ev in proposals:
        rec = {"issuer_name": h["issuer_name"], "isin_2010": h.get("isin_2010"),
               "replay_isin": i, "replay_rics": rics.get(i, []), "method": how, "evidence": ev,
               "cik_2010_lookup": next(hc for hh, hc in hside if hh is h),
               "cik_replay_lookup": rside[i], "prior_link": h.get("replay_isin"),
               "prior_link_method": h.get("link_method"),
               "sec_names_2010_cik": res.all_names(next(hc for hh, hc in hside if hh is h)["cik"])}
        (confirmed if hc_n[id(h)] == 1 and ic_n[i] == 1 else conflicts).append(rec)
    done_h = {r["issuer_name"] for r in confirmed}
    done_i = {r["replay_isin"] for r in confirmed}
    return {
        "confirmed": confirmed,
        "ambiguous": conflicts,
        "review_links_contradicted": [r for r in confirmed
                                      if r["prior_link"] and r["prior_link"] != r["replay_isin"]],
        "unconfirmed_holdings": [{"issuer_name": h["issuer_name"], "isin_2010": h.get("isin_2010"),
                                  "prior_link": h.get("replay_isin"), "cik_lookup": hc}
                                 for h, hc in hside if h["issuer_name"] not in done_h],
        "unconfirmed_replay_isins": [{"isin": i, "rics": rics.get(i, []), "cik_lookup": rc}
                                     for i, rc in rside.items() if i not in done_i],
    }


CSV_COLS = ["issuer_name", "shares", "value_usd", "cusip", "source_accession", "source_url",
            "fund", "as_of_date", "source_form", "raw_name", "implied_price", "issuer_cusip6",
            "isin_2010", "name_13f", "match_13f", "match_13f_detail", "replay_isin",
            "link_method", "lseg_ric", "lseg_close", "price_log_ratio", "sec_confirmation",
            "sec_evidence"]


def run(offline: bool = False) -> dict:
    cusip_names, starred, meta13f = load_13f(offline)
    idx13 = build_13f_index(cusip_names, starred)
    inventory = spy_filing_inventory(offline)
    rics = load_identifiers()
    lseg_names = load_lseg_names(rics)
    sessions = sorted({c[1] for c in COMPARISONS})
    replay_sets = load_replay_sets(sessions)
    all_rics = {r for s in replay_sets.values() for i in s for r in rics.get(i, [])}
    prices = load_prices(all_rics, sorted({c[2] for c in COMPARISONS if c[2]}))

    parsed: dict[str, list[dict]] = {}
    src_meta = []
    srcs = {s.key: s for s in SOURCES}
    for src in SOURCES:
        path = fetch(src.url, SEC_CACHE / src.accession / src.document, offline)
        doc = read_html(path)
        hs = parse_schedule(doc, src.fund_title)
        resolve_holdings(hs, src, idx13, cusip_names, starred)
        parsed[src.key] = hs
        total = find_total_common_stocks(doc, src.fund_title)
        src_meta.append({**asdict(src), "url": src.url, "index_url": src.index_url,
                         "sha256": sha256(path), "holding_lines": len(hs),
                         "sum_value_usd": sum(h["value_usd"] for h in hs),
                         "reported_total_common_stocks_usd": total})

    presence: dict[str, dict[str, bool]] = {}
    iss_by_src = {k: {h["issuer_cusip6"] for h in v if h["issuer_cusip6"]} for k, v in parsed.items()}
    for iss in set().union(*iss_by_src.values()) | {
            c[:6] for s in replay_sets.values() for i in s if (c := isin_embedded_cusip(i))}:
        presence[iss] = {k: iss in v for k, v in iss_by_src.items()}

    comps = {}
    resolver = SecResolver(offline)
    for key, session, psession in COMPARISONS:
        info = link(parsed[key], replay_sets[session], rics, lseg_names, prices, psession)
        summ = summarize(srcs[key], parsed[key], replay_sets[session], rics, lseg_names, idx13,
                         info, presence, session, psession)
        if psession:  # SEC confirmation for unlinked lines and review-tier links
            todo = [h for h in parsed[key] if not h["replay_isin"]
                    or h["link_method"] not in STRONG_METHODS]
            used = {h["replay_isin"] for h in parsed[key]
                    if h["replay_isin"] and h["link_method"] in STRONG_METHODS}
            summ["sec_confirmation"] = confirm_pairs(
                todo, sorted(replay_sets[session] - used), rics, cusip_names, resolver, lseg_names)
            apply_confirmations(parsed[key], summ["sec_confirmation"])
            summ["counts_after_sec_confirmation"] = after_counts(parsed[key], replay_sets[session])
        comps[f"{key}_vs_replay_{session}"] = summ

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for src in SOURCES:
            w.writerows(parsed[src.key])

    result = {
        "generated_by": "TrainingData/pit/spy_holdings_roster.py",
        "question": "Does the reverse-replayed 2011-01-03 opening roster match an independent "
                    "S&P 500 roster from SEC fund holdings at 2010-12-31?",
        "spy_filing_inventory": inventory,
        "sources": src_meta,
        "identifier_bridge_13f": meta13f,
        "price_tolerance_abs_log": PRICE_TOL,
        "replay_file": "TrainingData/indicators_data/lseg/20261119T132813190380Z/membership_dated_candidate.csv.gz",
        "membership_changes_2011H1": load_changes(),
        "comparisons": comps,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(render_markdown(result), encoding="utf-8")
    return result


def _md_escape(s) -> str:
    return str(s if s is not None else "").replace("|", "/")


def render_markdown(res: dict) -> str:
    """Short report. All numbers come from ``res``; interpretive sentences are marked."""
    src = {s["key"]: s for s in res["sources"]}
    comps = res["comparisons"]
    prim = comps[f"ivv_2010-12-31_vs_replay_{OPENING_SESSION}"]
    c = prim["counts"]
    inv = res["spy_filing_inventory"]
    L = []
    L.append("# S&P 500 opening roster check against SEC fund holdings (2010-12-31)\n")
    L.append("Generated by `TrainingData/pit/spy_holdings_roster.py`; full evidence in "
             "`research/spy_roster_2010_comparison.json`, parsed lines in "
             "`TrainingData/pit/overrides/spy_holdings_2010-12-31.csv`.\n")
    L.append("## Sources\n")
    L.append(f"SPY (CIK 884394) has filed {inv['n_q_filings_ever']} Form N-Q filings "
             f"([submissions index]({inv['url']})). Its full schedules are semiannual N-30D reports, "
             "so no SPY schedule is dated 2010-12-31. The exact-date primary source is IVV's N-Q, "
             "and the two SPY reports bracket it.\n")
    L.append("| Source | Form | As of | Accession | Lines | Parsed sum (USD) | Reported Total Common Stocks |")
    L.append("|---|---|---|---|---:|---:|---:|")
    for s in res["sources"]:
        L.append(f"| [{s['fund']}]({s['url']}) | {s['form']} | {s['as_of']} | {s['accession']} | "
                 f"{s['holding_lines']} | {s['sum_value_usd']:,} | "
                 f"{(s['reported_total_common_stocks_usd'] or 0):,} |")
    b = res["identifier_bridge_13f"]
    L.append(f"\nIssuer names are bridged to CUSIPs with the [SEC 13(f) list, Q4 2010]({b['url']}) "
             f"(current as of {b['list_current_as_of']}; {b['cusips_parsed']:,} non-option CUSIPs parsed). "
             "Each link is corroborated by the ratio of the LSEG unadjusted close to the fund's implied "
             f"price (value / shares); tolerance |log ratio| <= {res['price_tolerance_abs_log']}.\n")
    L.append("## Primary result: IVV 2010-12-31 vs replay 2011-01-03\n")
    L.append("LSEG's first post-2010 change is dated 2011-01-04 (`membership_changes_2011H1` in the JSON), "
             "so the 2010-12-31 close holdings and the 2011-01-03 opening roster should be the same index.\n")
    L.append("| Measure | Count |\n|---|---:|")
    for k, v in c.items():
        L.append(f"| {k} | {v if not isinstance(v, dict) else ', '.join(f'{a}: {n}' for a, n in v.items())} |")
    L.append(f"\nMaximum |log price ratio| among strong links: {prim['price_log_ratio_abs_max_strong']}.\n")
    if prim["strong_links_with_price_conflict"]:
        L.append("Strong (CUSIP) links whose price ratio exceeds the tolerance. The fund price is the "
                 "12-31 close and the LSEG price is the 01-03 close, so a one-day move is included:\n")
        L.append("| Holding | Implied 12-31 | LSEG RIC | LSEG 01-03 close | log ratio |\n|---|---:|---|---:|---:|")
        for h in prim["strong_links_with_price_conflict"]:
            L.append(f"| {_md_escape(h['issuer_name'])} | {h['implied_price']} | {h['lseg_ric']} | "
                     f"{h['lseg_close']} | {h['price_log_ratio']} |")
    if prim["strong_links_without_price"]:
        L.append("\nStrong links with no LSEG 2011-01-03 close on any RIC of the ISIN: " +
                 ", ".join(f"{_md_escape(h['issuer_name'])} ({h['replay_isin']})"
                           for h in prim["strong_links_without_price"]) + ".\n")
    L.append("\n### Links on name token plus price (review)\n")
    L.append("| Holding | Replay ISIN | RIC | log ratio |\n|---|---|---|---:|")
    for h in prim["links_needing_review"]:
        L.append(f"| {_md_escape(h['issuer_name'])} | {h['replay_isin']} | {h['lseg_ric']} | {h['price_log_ratio']} |")
    sc = prim.get("sec_confirmation")
    if sc:
        a = prim["counts_after_sec_confirmation"]
        L.append("\n## SEC confirmation of the remaining pairs\n")
        L.append(f"Before this step {c['holding_lines_unlinked']} IVV lines and "
                 f"{c['replay_isins_unlinked']} replay ISINs had no link, and {c['linked_needing_review']} "
                 "links rested on a name token plus price. Each was tested against SEC records only: "
                 "CUSIP -> issuer CIK from Schedule 13G/13D cover pages (EDGAR full-text search), "
                 "ticker -> CIK (company_tickers.json), name -> CIK (cik-lookup-data.txt), SEC name "
                 "histories (submissions JSON), the successor's 8-K12B/8-K12G3, or its Item 2.01 8-K "
                 "with the predecessor delisting (Form 25/15) within 45 days. Issuer names that funds "
                 "report beside the replay identifier in XML holdings tables are used only to find "
                 "the replay CIK. Only one-to-one results are accepted.\n")
        L.append("| After SEC confirmation | Count |\n|---|---:|")
        for k, v in a.items():
            L.append(f"| {k} | {v} |")
        L.append("\n| IVV line | 2010 ISIN | Replay ISIN | RIC | Evidence |\n|---|---|---|---|---|")
        for x in sorted(sc["confirmed"], key=lambda r: (r["method"], r["issuer_name"])):
            ev = x["evidence"]
            if x["method"] == "same_registrant":
                e = f"same CIK {ev['cik']}"
                if ev.get("via_fund_reported_name"):
                    e += f" (replay CIK via fund-reported name '{_md_escape(ev['via_fund_reported_name'])}')"
            else:
                e = (f"{ev['form']} {ev['accession']} ({ev['filing_date']}), CIK {ev['successor_cik']} "
                     f"names '{_md_escape(ev['predecessor_name_found'])}' (CIK {ev['predecessor_cik']})")
            L.append(f"| {_md_escape(x['issuer_name'])} | {x['isin_2010'] or '-'} | {x['replay_isin']} | "
                     f"{', '.join(x['replay_rics'])} | {e} |")
        L.append("\n### Not SEC-confirmed\n")
        L.append("Lines with a prior link keep it at review tier (first name token plus price); "
                 "the others are unlinked.\n")
        L.append("| IVV line | 2010 CIK found | Prior link |\n|---|---|---|")
        for x in sc["unconfirmed_holdings"]:
            L.append(f"| {_md_escape(x['issuer_name'])} | {x['cik_lookup'].get('cik') or '-'} | "
                     f"{x['prior_link'] or '-'} |")
        L.append("\n| Replay ISIN | RICs | LSEG names |\n|---|---|---|")
        lsn = {x["isin"]: x["lseg_names"] for x in prim["only_in_replay"]}
        for x in sc["unconfirmed_replay_isins"]:
            L.append(f"| {x['isin']} | {', '.join(x['rics']) or '-'} | "
                     f"{_md_escape('; '.join(lsn.get(x['isin'], [])[:2])) or '-'} |")
    L.append("\n## Bracketing SPY reports\n")
    L.append("| Comparison | Lines | Linked | Strong | SEC-confirmed | Unlinked lines | Unlinked replay ISINs |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for k, cc in comps.items():
        n = cc.get("counts_after_sec_confirmation")
        b = cc["counts"]
        if n:
            L.append(f"| {k} | {b['holding_lines']} | {n['linked_total']} | "
                     f"{n['linked_strong_cusip_or_name_price']} | {n['linked_sec_confirmed']} | "
                     f"{n['holding_lines_unlinked']} | {n['replay_isins_unlinked']} |")
        else:
            L.append(f"| {k} | {b['holding_lines']} | {b['linked_total']} | {b['linked_strong']} | - | "
                     f"{b['holding_lines_unlinked']} | {b['replay_isins_unlinked']} |")
    spy = comps.get("spy_2011-03-31_vs_replay_2011-03-31", {}).get("sec_confirmation")
    if spy:
        L.append("\nSPY 2011-03-31 lines still unlinked: " + ", ".join(
            _md_escape(x["issuer_name"]) for x in spy["unconfirmed_holdings"]) + ".")
    L.append("\nThe 2010-09-30 comparison has no LSEG prices (they start 2011-01-03), so its name links stay "
             "in review and it gets no SEC step. It is one quarter early and is reference only.\n")
    L.append(INTERPRETATION)
    return "\n".join(L) + "\n"


INTERPRETATION = """## Interpretation (analyst notes)

* Membership: no IVV 2010-12-31 line is evidenced as missing from the replay, and no replay ISIN
  as extra. Every confirmed link is one-to-one, and the residual on each side has the same size.
  The open IVV items are Tyco and Rowan, whose replay ISINs (CH0100383485, GB00B6SLMV12) are
  non-US and appear in no issuer filing found, and Cablevision NY Group, a tracking-stock line
  that keeps its review-tier link to US12686C1099 (Cablevision Systems).
* Identifier vintage: the replay holds the *current* LSEG ISIN for companies whose CUSIP/ISIN
  changed after 2010 through a rename, holding-company reorganization, redomicile or merger
  (for example Exxon US30231G1022 -> US30233Q1085, Google US38259P5089 -> Alphabet US02079K3059).
  A point-in-time join on ISIN to 2010-era data misses these names; it needs a dated ISIN map.
  The confirmed table above is that map for the opening roster.
* A replay ISIN that belongs to the successor carries the successor's later identity, not a
  2010 security. The holding-company cases (Alphabet, Walgreens Boots Alliance, Express Scripts
  Holding, ExxonMobil Holdings) are distinct registrants: use their ISINs for 2011 only through a
  dated mapping.
* SPY 2011-03-31 holds Edwards Lifesciences, which LSEG adds 2011-04-01 (index funds trade at the
  close before an effective date), and Huntington Ingalls, a Northrop Grumman spin-off distributed
  2011-03-31 that is not an S&P 500 constituent. Neither is a replay error.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="use cached SEC files only")
    a = ap.parse_args(argv)
    res = run(offline=a.offline)
    for s in res["sources"]:
        print(s["key"], s["holding_lines"], s["sum_value_usd"], s["reported_total_common_stocks_usd"])
    for k, c in res["comparisons"].items():
        print(k, json.dumps(c["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
