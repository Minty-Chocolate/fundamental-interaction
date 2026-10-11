"""Optional group-label panels for generic cross-sectional scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


# The SIC divisions supplied for this project.  Unused ranges such as
# 1800–1999 intentionally have no label and therefore cannot be group-scored.
_SIC_DIVISIONS: tuple[tuple[int, int, str], ...] = (
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


def sic_division_panel(sic: pd.DataFrame) -> pd.DataFrame:
    """Map a daily SIC-code panel into the project's 11 SIC divisions.

    The returned panel is only a generic ``groups`` input to :func:`score`.
    It has no effect on global winsorization or final gross/net normalization.
    """
    values = sic.to_numpy(dtype=float, na_value=np.nan)
    conditions = [(values >= lower) & (values <= upper) for lower, upper, _ in _SIC_DIVISIONS]
    labels = [label for _, _, label in _SIC_DIVISIONS]
    grouped = np.select(conditions, labels, default=None)
    return pd.DataFrame(grouped, index=sic.index, columns=sic.columns, dtype="string").replace("None", pd.NA)
