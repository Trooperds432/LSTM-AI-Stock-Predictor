"""Tests for the PIT layer: confirmation guard, override validation, roster mapping, security master."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TrainingData"))

from pit import config, identity, membership, overrides  # noqa: E402

DATA = config.OUTPUT / "membership_pit.csv"
OLD = config.SNAPSHOT["main"] / "membership_dated_candidate.csv.gz"


def test_guard_raises_in_confirmation_period():
    config.guard_confirmation(["2024-12-31"])  # fine
    with pytest.raises(config.ConfirmationPeriodError):
        config.guard_confirmation(["2024-12-31", "2025-01-02"])
    config.guard_confirmation(["2025-06-01"], confirm=True)  # explicit final run allowed


def test_research_window_drops_confirmation_rows():
    df = pd.DataFrame({"session": ["2024-12-31", "2025-01-02", "2026-01-05"], "x": [1, 2, 3]})
    assert list(config.research_window(df)["x"]) == [1]
    assert len(config.research_window(df, confirm=True)) == 3


def _write(tmp_path, name, text):
    (tmp_path / f"{name}.csv").write_text(text)


def test_override_loader_rejects_missing_source(tmp_path):
    _write(tmp_path, "membership_alias_dedup", "keep_ric,drop_ric,before_date,reason,source\nA.N,A.OQ,2020-01-01,x,\n")
    with pytest.raises(overrides.OverrideError, match="without a source"):
        overrides.load("membership_alias_dedup", tmp_path)


def test_override_loader_rejects_duplicates_and_bad_change(tmp_path):
    _write(tmp_path, "extra_membership_events",
           "date,isin,change,reason,source\n2020-01-02,X,Joiner,r,s\n2020-01-02,X,Joiner,r,s\n")
    with pytest.raises(overrides.OverrideError, match="duplicate"):
        overrides.load("extra_membership_events", tmp_path)
    _write(tmp_path, "extra_membership_events", "date,isin,change,reason,source\n2020-01-02,X,Added,r,s\n")
    with pytest.raises(overrides.OverrideError, match="Joiner or Leaver"):
        overrides.load("extra_membership_events", tmp_path)


def test_real_override_tables_are_valid_and_fully_sourced():
    for name in overrides.SCHEMAS:
        overrides.load(name)
    assert overrides.pending_sources() == []


def test_map_roster_applies_name_override_dedup_and_substitution():
    ids = {"DD.N": "US_DD_NEW", "DOW.N^I17": "US_DOW_OLD", "HST.N": "US_HST", "HST.OQ": "US_HST",
           "JCI.N": "IE_JCI_COMBINED"}
    names = pd.DataFrame([{"constituent_ric": "DD.N", "constituent_name": "DOW CHEMICAL ORD", "instrument": "DOW.N^I17"}])
    dedup = pd.DataFrame([{"keep_ric": "HST.N", "drop_ric": "HST.OQ", "before_date": "2020-11-02"}])
    subs = pd.DataFrame([{"from_isin": "IE_JCI_COMBINED", "to_isin": "IE_TYCO", "valid_from": "2014-11-17",
                          "valid_to": "2016-09-06", "applies_to": "checkpoints"}])
    rows = pd.DataFrame({"Constituent RIC": ["DD.N", "DD.N", "HST.N", "HST.OQ", "JCI.N"],
                         "Constituent Name": ["DOW CHEMICAL ORD", "DUPONT", "HOST", "HOST", "TYCO"]})
    got = membership.map_roster(rows, "2015-12-31", ids, names, subs, dedup)
    assert got == {"US_DOW_OLD", "US_DD_NEW", "US_HST", "IE_TYCO"}
    with pytest.raises(ValueError, match="collision"):  # without the name override DD.N appears twice
        membership.map_roster(rows, "2015-12-31", ids, names.iloc[0:0], subs, dedup)


def test_security_master_validation_detects_overlaps():
    sm = pd.DataFrame([
        ("I1", "A.N", "2011-01-03", "2099-12-31", "primary", "s"),
        ("I1", "A.OQ", "2015-01-01", "2099-12-31", "primary", "s"),   # overlapping primary for I1
        ("I2", "B.N", "2011-01-03", "2016-01-01", "primary", "s"),
        ("I3", "B.N", "2015-06-01", "2099-12-31", "primary", "s"),    # B.N maps to two securities in 2015
    ], columns=["isin", "ric", "valid_from", "valid_to", "role", "source"])
    v = identity.validate(sm)
    assert not v["pass"] and "I1" in v["primary_overlaps"] and "B.N" in v["ric_overlaps"]
    assert identity.primary_ric_on(sm, "I2", "2015-12-31") == "B.N"
    assert identity.primary_ric_on(sm, "I2", "2016-01-01") is None


@pytest.mark.skipif(not (DATA.exists() and OLD.exists()), reason="LSEG data not present")
def test_ported_replay_reproduces_original_exactly():
    assert pd.read_csv(DATA).equals(pd.read_csv(OLD))
