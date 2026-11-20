"""Tests for LSEG deal-synopsis term parsing used by pit.settlement."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TrainingData"))
from pit.settlement import parse_terms  # noqa: E402


def test_parses_mixed_cash_and_ratio():
    t = "Chubb shareholders received USD 62.93 in cash and 0.6019 ACE ordinary shares for each share held."
    ratios, cash = parse_terms(t)
    assert ratios == [0.6019] and cash == [62.93]


def test_parses_stock_only_with_company_name_between():
    ratios, cash = parse_terms("offered 1.46 ConocoPhillips common shares per Concho share")
    assert ratios == [1.46] and cash == []


def test_ignores_large_numbers_and_share_counts():
    ratios, _ = parse_terms("acquired 25 mil common shares, or 1.2 ABC shares per XYZ share")
    assert ratios == [1.2]


def test_collar_value_terms_yield_no_ratio():
    ratios, cash = parse_terms("AT&T offered USD 28.5 cash and USD 66.5 in AT&T common shares for every DirecTV share")
    assert ratios == [] and cash == [28.5]
