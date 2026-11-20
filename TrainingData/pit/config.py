"""Paths, run-folder registry and the confirmation-period guard for the PIT data layer."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TRAINING_DATA = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_DATA.parent
LSEG_ROOT = TRAINING_DATA / "indicators_data" / "lseg"
OVERRIDES = Path(__file__).resolve().parent / "overrides"
OUTPUT = TRAINING_DATA / "indicators_data" / "pit"   # gitignored with the rest of indicators_data/

# Every LSEG run folder the layer reads. Paths are relative to LSEG_ROOT.
SNAPSHOT: dict[str, Path] = {
    "main": LSEG_ROOT / "20261119T132813190380Z",
    "actions": LSEG_ROOT / "actions" / "20261119T153008700524Z",
    "recovery": LSEG_ROOT / "recovery" / "20261119T151537670853Z",
    "bemis": LSEG_ROOT / "recovery" / "20261119T152440659949Z",
    "gap_recovery": LSEG_ROOT / "gap_recovery" / "20261119T153420647171Z",
    "successors": LSEG_ROOT / "successors" / "20261119T173231936867Z",
    "micro_focus": LSEG_ROOT / "micro_focus" / "20261119T173917164993Z",
    "tyco": LSEG_ROOT / "tyco" / "20261119T174605114045Z",
    "baker": LSEG_ROOT / "baker" / "20261119T175021011596Z",
}

START = "2011-01-03"
END = "2026-11-18"
CONFIRM_START = pd.Timestamp("2025-01-02")  # reserved confirmation period (preregistered)


class ConfirmationPeriodError(RuntimeError):
    """Raised when research code touches the reserved confirmation period without confirm=True."""


def guard_confirmation(dates, confirm: bool = False, what: str = "data") -> None:
    """Raise if any date is in the reserved confirmation period and confirm is False."""
    if confirm:
        return
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).dropna().unique()))
    if len(d) and d.max() >= CONFIRM_START:
        raise ConfirmationPeriodError(
            f"{what} reaches {d.max().date()} which is inside the reserved confirmation period "
            f"(>= {CONFIRM_START.date()}). Pass confirm=True only for the single final confirmation run."
        )


def research_window(df: pd.DataFrame, date_col: str = "session", confirm: bool = False) -> pd.DataFrame:
    """Return rows usable for research: drops the confirmation period unless confirm=True."""
    if confirm:
        return df
    return df.loc[pd.to_datetime(df[date_col]) < CONFIRM_START]
