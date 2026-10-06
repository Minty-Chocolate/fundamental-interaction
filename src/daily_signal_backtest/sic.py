"""SIC classification for wide date-by-permno research panels."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Codes deliberately follow the user's SIC division table.  Intervals omitted
# here (1800-1999 and 9000-9099, for example) map to NA and cannot be scored
# under SIC_LEVEL=1.
DIVISIONS: tuple[tuple[int, int, str], ...] = (
    (100, 999, "agriculture_forestry_fishing"),
    (1000, 1499, "mining"),
    (1500, 1799, "construction"),
    (2000, 3999, "manufacturing"),
    (4000, 4999, "transportation_communications_utilities"),
    (5000, 5199, "wholesale_trade"),
    (5200, 5999, "retail_trade"),
    (6000, 6799, "finance_insurance_real_estate"),
    (7000, 8999, "services"),
    (9100, 9729, "public_administration"),
    (9900, 9999, "nonclassifiable"),
)


def division_codes(sic: np.ndarray) -> np.ndarray:
    """Return compact 0--10 division codes, or -1 for unmapped SIC values."""
    conditions = [(sic >= low) & (sic <= high) for low, high, _ in DIVISIONS]
    return np.select(conditions, np.arange(len(DIVISIONS), dtype=np.int8), default=-1).astype(np.int8, copy=False)


def division_panel(sic: pd.DataFrame) -> pd.DataFrame:
    """Map a SIC wide panel to same-shaped SIC-division labels.

    This is fully vectorized across both axes.  It performs no security/date
    loop and preserves missing/unused SIC ranges as ``pd.NA``.
    """
    values = sic.to_numpy(dtype=float, na_value=np.nan)
    conditions = [(values >= low) & (values <= high) for low, high, _ in DIVISIONS]
    labels = [label for _, _, label in DIVISIONS]
    mapped = np.select(conditions, labels, default=None)
    return pd.DataFrame(mapped, index=sic.index, columns=sic.columns, dtype="string").replace("None", pd.NA)


def score_group_panel(sic: pd.DataFrame, sic_level: int | None) -> pd.DataFrame | None:
    """Return the SIC grouping panel requested by the backtest setting."""
    if sic_level is None:
        return None
    if sic_level != 1:
        raise ValueError("Only sic_level=None or sic_level=1 is configured")
    return division_panel(sic)
