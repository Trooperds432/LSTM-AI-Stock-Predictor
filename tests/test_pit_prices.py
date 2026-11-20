"""Tests for pit.prices: bar validity, dated RIC->security mapping, alias selection."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TrainingData"))
from pit import prices  # noqa: E402


def bars(ric, sessions, close=10.0, **kw):
    df = pd.DataFrame({"session": sessions, "ric": ric, "open": close, "high": close + 1, "low": close - 1,
                       "close": close, "volume_venue": 100.0, "source_file": "f"})
    for k, v in kw.items():
        df[k] = v
    df["valid_bar"] = prices.valid_mask(df)
    return df


def test_valid_mask_rejects_inconsistent_bars():
    df = pd.DataFrame({"open": [10, 10, 10, 10], "high": [11, 9, 11, 11], "low": [9, 9, 0, 9],
                       "close": [10, 10, 10, 10], "volume_venue": [5, 5, 5, 0]})
    assert list(prices.valid_mask(df)) == [True, False, False, False]


def test_dated_ric_maps_to_successive_securities():
    # One RIC carries security A before 2016-09-06 and security B from then on (Tyco -> JCI pattern)
    sm = pd.DataFrame([("A", "JCI.N", "2011-01-03", "2016-09-06", "primary", "s"),
                       ("B", "JCI.N", "2016-09-06", "2099-12-31", "primary", "s")],
                      columns=["isin", "ric", "valid_from", "valid_to", "role", "source"])
    px = prices.map_to_securities(bars("JCI.N", ["2016-09-02", "2016-09-06", "2016-09-07"]), sm)
    assert dict(zip(px.session, px["isin"])) == {"2016-09-02": "A", "2016-09-06": "B", "2016-09-07": "B"}


def test_alias_rows_are_never_used_as_prices():
    sm = pd.DataFrame([("X", "CPB.OQ", "2011-01-03", "2099-12-31", "primary", "s"),
                       ("X", "CPB.N", "2011-01-03", "2099-12-31", "alias", "s")],
                      columns=["isin", "ric", "valid_from", "valid_to", "role", "source"])
    b = pd.concat([bars("CPB.OQ", ["2024-08-19"], close=50.0), bars("CPB.N", ["2024-08-19"], close=49.97)])
    px = prices.map_to_securities(b, sm)
    assert list(px.ric) == ["CPB.OQ"] and list(px.close) == [50.0]


def test_two_primary_bars_for_one_security_session_raises():
    sm = pd.DataFrame([("X", "A.N", "2011-01-03", "2099-12-31", "primary", "s"),
                       ("X", "A.OQ", "2011-01-03", "2099-12-31", "primary", "s")],
                      columns=["isin", "ric", "valid_from", "valid_to", "role", "source"])
    b = pd.concat([bars("A.N", ["2020-01-02"]), bars("A.OQ", ["2020-01-02"])])
    with pytest.raises(ValueError, match="two primary bars"):
        prices.map_to_securities(b, sm)
