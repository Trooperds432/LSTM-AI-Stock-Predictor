"""Unit tests for the SEC holdings roster builder (no network, small fixtures)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TrainingData" / "pit"))

import spy_holdings_roster as s  # noqa: E402


# ----------------------------------------------------------------- CUSIP / ISIN check digits
@pytest.mark.parametrize("cusip8,check", [
    ("03783310", 0),   # Apple
    ("30231G10", 2),   # Exxon Mobil (2010)
    ("19121610", 0),   # Coca-Cola
    ("G9829010", 2),   # XL Group plc (CINS, 13F list line 'G98290 10 2')
    ("H0023R10", 5),   # ACE Ltd (13F list line 'H0023R 10 5')
    ("38259P50", 8),   # Google class A (13F list line '38259P 50 8')
])
def test_cusip_check_digit(cusip8, check):
    assert s.cusip_check_digit(cusip8) == check
    assert s.is_valid_cusip(cusip8 + str(check))
    assert not s.is_valid_cusip(cusip8 + str((check + 1) % 10))


@pytest.mark.parametrize("cusip,isin", [
    ("037833100", "US0378331005"),
    ("30231G102", "US30231G1022"),
    ("191216100", "US1912161007"),
    ("38259P508", "US38259P5089"),
])
def test_cusip_to_isin(cusip, isin):
    assert s.cusip_to_isin(cusip) == isin
    assert s.is_valid_isin(isin)


def test_cusip_to_isin_rejects_bad_cusip():
    with pytest.raises(ValueError):
        s.cusip_to_isin("037833101")


def test_isin_embedded_cusip():
    assert s.isin_embedded_cusip("US0378331005") == "037833100"
    assert s.isin_embedded_cusip("AN8068571086") == "806857108"  # Schlumberger, CUSIP-based
    assert s.isin_embedded_cusip("CH0044328745") is None         # Swiss ISIN, not CUSIP-based
    assert s.isin_embedded_cusip("US0378331006") is None         # bad ISIN check digit


# ----------------------------------------------------------------- schedule parsing
def _row(name, shares="", value="", dollar=False):
    return (f"<TR><TD><DIV>{name}</DIV></TD><TD>&nbsp;</TD><TD>&nbsp;</TD>"
            f"<TD align=right>{shares}</TD><TD>&nbsp;</TD><TD>&nbsp;</TD>"
            f"<TD>{'$' if dollar else '&nbsp;'}</TD><TD align=right>{value}</TD></TR>")


SPY_FIXTURE = (
    "<HTML><BODY><P>Schedule of Investments</P><P>September&#160;30, 2010</P><TABLE>"
    + _row("<B>Common Stocks</B>", "<B>Shares</B>", "<B>Value</B>")
    + _row("3M Co.&#160;", "1,000", "86,710", dollar=True)
    + _row("Total System Services, Inc.", "1,244,000", "18,958,804")
    + _row("State Street Corp. (b)", "3,793,465", "142,861,892")
    + "</TABLE><P>Schedule of Investments (continued)</P><TABLE>"
    + _row("Coca-Cola Co.", "200", "11,704")
    + _row("Total Common Stocks (b) (Cost $1)", "", "161,919,110")
    + _row("Something after total", "1", "2")
    + "</TABLE></BODY></HTML>"
)


def test_parse_spy_style_schedule():
    rows = s.parse_schedule(SPY_FIXTURE)
    assert [r["issuer_name"] for r in rows] == [
        "3M Co.", "Total System Services, Inc.", "State Street Corp.", "Coca-Cola Co."]
    assert rows[0]["shares"] == 1000 and rows[0]["value_usd"] == 86710
    # A company named "Total ..." is a holding; the Total Common Stocks line is not.
    assert sum(r["value_usd"] for r in rows) == s.find_total_common_stocks(SPY_FIXTURE)


def _ishares_page(title, body_rows):
    return ("<P>Schedule of Investments (Unaudited)</P><P>iSHARES<SUP>&reg;</SUP> "
            f"{title} </P><P>December 31, 2010</P><TABLE>" + "".join(body_rows) + "</TABLE>")


ISHARES_FIXTURE = "<HTML>" + "".join([
    _ishares_page("S&amp;P 500 GROWTH INDEX FUND", [
        _row("COMMON STOCKS&#151;99.9%"), _row("Apple Inc.(a)", "10", "3,226")]),
    _ishares_page("S&amp;P 500 INDEX FUND", [
        _row("COMMON STOCKS&#151;99.79%"), _row("ADVERTISING&#151;0.17%"),
        _row("Interpublic Group of Companies Inc. (The)(a)", "1,102,339", "11,706,840", True),
        _row("Omnicom Group Inc.", "678,057", "31,055,011"),
        _row("", "", "42,761,851")]),  # industry subtotal: value only
    _ishares_page("S&amp;P 500 INDEX FUND (Continued)", [
        _row("Xerox Corp.", "100", "1,152"),
        _row("TOTAL COMMON STOCKS"), _row("(Cost: $9)", "", "42,763,003"),
        _row("SHORT-TERM INVESTMENTS&#151;1.5%"),
        _row("BlackRock Cash Funds: Institutional, SL Agency Shares(c)(d)", "5,000", "5,000")]),
    _ishares_page("S&amp;P MIDCAP 400 INDEX FUND", [
        _row("COMMON STOCKS&#151;99%"), _row("Acme Corp.", "5", "50")]),
]) + "</HTML>"


def test_parse_ishares_fund_pages_only():
    rows = s.parse_schedule(ISHARES_FIXTURE, "S&P 500 INDEX FUND")
    names = [r["issuer_name"] for r in rows]
    assert names == ["Interpublic Group of Companies Inc. (The)", "Omnicom Group Inc.", "Xerox Corp."]
    assert rows[0]["raw_name"].endswith("(a)")  # footnote kept in raw_name, stripped in issuer_name
    assert s.find_total_common_stocks(ISHARES_FIXTURE, "S&P 500 INDEX FUND") == 42_763_003
    assert sum(r["value_usd"] for r in rows) == 42_763_003


def test_strip_footnotes():
    assert s.strip_footnotes("Foo Inc. (a)(b)") == "Foo Inc."
    assert s.strip_footnotes("Bar Co. (The)") == "Bar Co. (The)"
    assert s.strip_footnotes("Baz Corp. (a,b)") == "Baz Corp."


# ----------------------------------------------------------------- 13F list and names
THIRTEENF_FIXTURE = """\
CUSIP NO       ISSUER NAME                   ISSUER DESCRIPTION STATUS
191216 10 0 *  COCA COLA CO                  COM
191216 90 0    COCA COLA CO                  CALL
191216 95 0    COCA COLA CO                  PUT
254687 10 6 *  DISNEY WALT CO                COM DISNEY
65248E 10 4 *  NEWS CORP                     COM
65248E 20 3 *  NEWS CORP
681919 AV 8    OMNICOM GROUP INC            NOTE
681919 10 6 *  OMNICOM GROUP INC
682134 10 1    OMNICOM GROUP INC
469814 10 7 *  JACOBS ENGR GROUP INC DEL     COM
50540R 40 9 *  LABORATORY CORP AMER HLDGS    COM NEW
Run Date: 1/5/2011  ** List of Section 13F Securities **      Page 106
"""


def test_parse_13f_text():
    names, starred = s.parse_13f_text(THIRTEENF_FIXTURE)
    assert names["191216100"] == "COCA COLA CO"
    assert "191216900" not in names and "191216950" not in names  # option series dropped
    assert "681919AV8" in names  # debt kept in the list but not chosen as equity
    assert "191216100" in starred and "682134101" not in starred


@pytest.fixture()
def idx():
    names, starred = s.parse_13f_text(THIRTEENF_FIXTURE)
    return s.build_13f_index(names, starred), names, starred


def test_name_match_exact_and_reordered(idx):
    index, _, _ = idx
    assert index.match("Coca-Cola Co.").key == "191216"
    assert index.match("Walt Disney Co. (The)").key == "254687"  # token bag handles word order


def test_name_match_abbreviations(idx):
    index, _, _ = idx
    m = index.match("Jacobs Engineering Group Inc.")
    assert m.key == "469814" and m.method == "aligned_name"
    assert index.match("Laboratory Corp. of America Holdings").key == "50540R"


def test_name_match_tiebreak_and_unmatched(idx):
    index, _, _ = idx
    m = index.match("Omnicom Group Inc.")
    assert m.key == "681919" and m.method.endswith("optioned_tiebreak")
    assert index.match("Nonexistent Widgets Inc.").key is None


def test_equity_cusip_resolution(idx):
    _, names, starred = idx
    assert s.equity_cusip("191216", names, starred, None)[0] == "191216100"
    assert s.equity_cusip("681919", names, starred, None)[0] == "681919106"  # debt ignored
    # Two optioned equity classes: a class-labelled holding is not guessed.
    assert s.equity_cusip("65248E", names, starred, "A")[0] is None


# ----------------------------------------------------------------- linking
def test_link_cusip_then_lseg_name_with_price():
    names, starred = s.parse_13f_text(THIRTEENF_FIXTURE + "30231G 10 2 *  EXXON MOBIL CORP   COM\n")
    index = s.build_13f_index(names, starred)
    src = s.SOURCES[0]
    holdings = [
        {"issuer_name": "Coca-Cola Co.", "raw_name": "Coca-Cola Co.", "shares": 10, "value_usd": 650},
        {"issuer_name": "Exxon Mobil Corp.", "raw_name": "Exxon Mobil Corp.", "shares": 10, "value_usd": 731},
        {"issuer_name": "Omnicom Group Inc.", "raw_name": "Omnicom Group Inc.", "shares": 10, "value_usd": 458},
    ]
    s.resolve_holdings(holdings, src, index, names, starred)
    replay = {"US1912161007", "US30233Q1085", "US9999999990"}
    rics = {"US1912161007": ["KO.N"], "US30233Q1085": ["XOM.N"], "US9999999990": ["ZZZ.N"]}
    lseg_names = {"US30233Q1085": ["EXXON MOBIL ORD"], "US9999999990": ["UNRELATED ORD"]}
    prices = {("KO.N", "D"): 65.5, ("XOM.N", "D"): 74.5, ("ZZZ.N", "D"): 46.0}
    s.link(holdings, replay, rics, lseg_names, prices, "D")
    ko, xom, omc = holdings
    assert (ko["replay_isin"], ko["link_method"]) == ("US1912161007", "cusip_issuer")
    assert ko["isin_2010"] == "US1912161007"
    # Successor ISIN: no shared CUSIP, linked by LSEG name and corroborated by price.
    assert (xom["replay_isin"], xom["link_method"]) == ("US30233Q1085", "lseg_name+price")
    assert xom["isin_2010"] == "US30231G1022"
    # No name evidence: never linked on price alone.
    assert omc["replay_isin"] is None


# ----------------------------------------------------------------- SEC confirmation
def _hit(acc, ciks, names, form="SC 13G"):
    return {"_id": f"{acc}:doc.htm", "_source": {"ciks": ciks, "display_names": names, "form": form}}


def test_subject_from_hits_dominance():
    parent, sub = ("0001166126", "J C PENNEY CO INC"), ("0000077182", "J C PENNEY CORP INC")
    # 7 of 10 hits name the parent; the filer (Vanguard) co-appears on only two of them.
    hits = [_hit(f"a-{i}", [parent[0]] + (["0000102909"] if i < 2 else []),
                 [parent[1]] + (["VANGUARD"] if i < 2 else [])) for i in range(7)]
    hits += [_hit("b-1", [sub[0]], [sub[1]]), _hit("b-2", [sub[0]], [sub[1]])]
    hits += [_hit("c-1", ["0000000001"], ["OTHER"])]
    cik, name, n, acc = s.subject_from_hits(hits)
    assert cik == "1166126" and n == 10 and acc  # 70% of hits, > 2x the runner-up
    # Two CIKs equally present: no subject.
    tie = [_hit(f"t-{i}", ["0000000001", "0000000002"], ["A", "B"]) for i in range(4)]
    assert s.subject_from_hits(tie)[0] is None
    assert s.subject_from_hits([])[0] is None


def test_record_issuer_name_is_record_scoped():
    xml = ("<informationTable>"
           "<infoTable><nameOfIssuer>TWO HBRS INVT CORP</nameOfIssuer><cusip>90187B408</cusip></infoTable>"
           "<infoTable><nameOfIssuer>RS LEGACY CORP</nameOfIssuer><cusip>74979E101</cusip></infoTable>"
           "</informationTable>")
    assert s.record_issuer_name(xml, "74979E101") == "RS LEGACY CORP"
    # An identifier outside any record yields nothing, never a neighbouring record's name.
    assert s.record_issuer_name(xml + "<x>IE00BY9D5467</x>", "IE00BY9D5467") is None


def test_normalize_handles_fund_and_edgar_name_forms():
    assert s.normalize_name("RS Legacy Corp (formerly Radio") == "RS LEGACY"
    assert s.normalize_name("R S Legacy Corporati") == "R S LEGACY"
    assert s.normalize_name("JDS UNIPHASE CORP /CA/") == "JDS UNIPHASE"
    assert s.names_compatible("R S Legacy Corporati", ["RS Legacy Corp", "RADIOSHACK CORP"])
    assert s.names_compatible("Sara Lee Corp.", ["Hillshire Brands Co", "Sara Lee Corp", "LEE SARA CORP"])


def test_distinctive_name():
    assert not s.distinctive_name("NEWS CORP")      # 'NEWS': too common to count as a mention
    assert not s.distinctive_name("ITT CORP")
    assert s.distinctive_name("GOOGLE INC.")
    assert s.distinctive_name("Actavis, Inc.")
    assert s.distinctive_name("WALGREEN CO")


class FakeResolver:
    """Stands in for SecResolver: CUSIP/ticker/name -> CIK, SEC names, succession filings."""

    def __init__(self, cusips, names, succession, texts, tickers=None):
        self.cusips, self.names, self.succ, self.texts = cusips, names, succession, texts
        self.tickers = tickers or {}

    def cusip_cik(self, cusip):
        return {"query": cusip, "cik": self.cusips.get(cusip), "display_name": ""}

    def ticker_cik(self, t):
        return self.tickers.get(t)

    def name_ciks(self, name):
        return {"query": name, "cik": None, "candidates": [], "periodic_filers": []}

    def all_names(self, cik):
        return self.names[cik]

    def submissions(self, cik):
        return {"filings": [], "former_names": [], "name": self.names[cik][0], "tickers": []}

    def succession_filings(self, cik):
        return self.succ.get(cik, [])

    def filing_text(self, cik, f):
        return self.texts[f["accessionNumber"]]

    def id_cited_by(self, ident):
        return []

    def fund_reported_names(self, ident, max_docs=4):
        return []


def test_confirm_pairs_same_registrant_and_succession():
    holdings = [
        {"issuer_name": "United Technologies Corp.", "cusip": "913017109", "isin_2010": "US9130171096",
         "issuer_cusip6": "913017"},
        {"issuer_name": "Google Inc. Class A", "cusip": "38259P508", "isin_2010": "US38259P5089",
         "issuer_cusip6": "38259P"},
        {"issuer_name": "Unrelated Widgets Inc.", "cusip": "999999999", "isin_2010": None,
         "issuer_cusip6": "999999"},
    ]
    res = FakeResolver(
        cusips={"913017109": "101829", "75513E101": "101829", "38259P508": "1288776",
                "02079K305": "1652044"},
        names={"101829": ["RTX Corp", "UNITED TECHNOLOGIES CORP"], "1288776": ["GOOGLE LLC", "GOOGLE INC."],
               "1652044": ["Alphabet Inc."]},
        succession={"1652044": [{"form": "8-K12B", "accessionNumber": "0001193125-15-336577",
                                 "filingDate": "2015-10-02", "primaryDocument": "d.htm"}]},
        texts={"0001193125-15-336577": "<p>Alphabet Inc. is the successor issuer to Google Inc.</p>"})
    out = s.confirm_pairs(holdings, ["US75513E1010", "US02079K3059"], {}, {}, res)
    got = {r["issuer_name"]: (r["replay_isin"], r["method"]) for r in out["confirmed"]}
    assert got["United Technologies Corp."] == ("US75513E1010", "same_registrant")
    assert got["Google Inc. Class A"] == ("US02079K3059", "succession_filing")
    assert [h["issuer_name"] for h in out["unconfirmed_holdings"]] == ["Unrelated Widgets Inc."]
    assert out["ambiguous"] == []
