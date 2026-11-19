"""Point-in-time tests for alternative data: insider and congressional features may only appear
on rows after the information was public."""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TrainingData"))
sys.path.insert(0, str(ROOT / "TrainingData" / "featuresPy"))
# insiderbuying imports sec_edgar_downloader at module level; stub it (no network in tests)
sys.modules.setdefault("sec_edgar_downloader", types.SimpleNamespace(Downloader=object))

import processor  # noqa: E402
import insiderbuying  # noqa: E402

TRADING = pd.bdate_range("2021-01-04", "2021-04-30")


def test_snap_to_next_trading_day():
    out = processor.snap_to_trading_days(["2021-01-09", "2021-01-11", "2025-01-01"], TRADING)  # Sat, Mon, beyond
    assert out[0] == pd.Timestamp("2021-01-11")
    assert out[1] == pd.Timestamp("2021-01-11")
    assert pd.isna(out[2])


def test_legacy_insider_shifted_and_sales_signed(tmp_path):
    f = tmp_path / "X_insider_trades_daily.csv"
    pd.DataFrame({"date": ["2021-01-05", "2021-01-05"], "shares": [100.0, 300.0],
                  "amount": [1000.0, 3000.0], "buy_flag": [1, 0]}).to_csv(f, index=False)
    g = processor.safe_read_insider(str(f), trading_dates=TRADING)
    # transaction Tue 2021-01-05 -> +2 BDays = Thu 2021-01-07; net = +100 - 300 = -200 (net selling)
    assert list(g["date"]) == [pd.Timestamp("2021-01-07")]
    assert g["insider_shares"].iloc[0] == -200
    assert g["insider_amount"].iloc[0] == -2000
    assert g["insider_buy_flag"].iloc[0] == 0


def test_filing_based_insider_not_shifted(tmp_path):
    f = tmp_path / "X_insider_trades_daily.csv"
    pd.DataFrame({"date": ["2021-01-08"], "shares": [50.0], "amount": [500.0], "buy_flag": [1],
                  "date_basis": ["filing_available"]}).to_csv(f, index=False)
    g = processor.safe_read_insider(str(f), trading_dates=TRADING)
    assert list(g["date"]) == [pd.Timestamp("2021-01-08")]
    assert g["insider_buy_flag"].iloc[0] == 1


def test_political_features_lagged_to_disclosure_deadline(tmp_path, monkeypatch):
    f = tmp_path / "X_political_trades.csv"
    pd.DataFrame({"transaction_date": ["2021-01-05"], "amount": ["$1,001 - $15,000"],
                  "transaction_type": ["Purchase"], "asset_type": ["Stock"], "senator": ["A"]}).to_csv(f, index=False)
    monkeypatch.setattr(processor, "_resolve_political_trades_csv_path", lambda t: str(f))
    agg = processor.build_political_daily_aggregates("X", trading_dates=TRADING)
    # 2021-01-05 + 45 days = Fri 2021-02-19 (a trading day)
    assert list(agg["date"]) == [pd.Timestamp("2021-02-19")]
    df = processor.merge_political_daily_features(pd.DataFrame({"date": TRADING}), "X")
    assert df.loc[df["date"] < "2021-02-19", "polit_trade_count"].sum() == 0
    assert df.loc[df["date"] == "2021-02-19", "polit_trade_count"].iloc[0] == 1


@pytest.mark.parametrize("stamp,expected", [
    ("20210212103000", "2021-02-12"),   # accepted intraday -> same day
    ("20210212173015", "2021-02-15"),   # accepted Friday after close -> next business day (Mon)
])
def test_form4_availability_date(tmp_path, stamp, expected):
    f = tmp_path / "full-submission.txt"
    f.write_text(f"<SEC-HEADER>\n<ACCEPTANCE-DATETIME>{stamp}\nFILED AS OF DATE:\t20210212\n</SEC-HEADER>\n")
    assert insiderbuying.availability_date_from_submission(str(f)) == expected


def test_form4_parse_uses_availability_date():
    xml = ("<ownershipDocument><nonDerivativeTable><nonDerivativeTransaction>"
           "<transactionDate><value>2021-02-01</value></transactionDate>"
           "<transactionCoding><transactionCode>P</transactionCode></transactionCoding>"
           "<transactionAmounts><transactionShares><value>10</value></transactionShares>"
           "<transactionPricePerShare><value>5</value></transactionPricePerShare></transactionAmounts>"
           "</nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>")
    rows = insiderbuying.parse_f4(xml, available_date="2021-02-03")
    assert rows == [("2021-02-03", 10.0, 50.0, 1)]
