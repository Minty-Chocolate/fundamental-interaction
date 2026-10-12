"""Optional group-label panels for generic cross-sectional scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


# The SIC divisions supplied for this project.  Unused ranges such as
# 1800–1999 intentionally have no label and therefore cannot be group-scored.
_SIC_DIVISIONS: tuple[tuple[int, int, int], ...] = (
    (100, 999, 1),
    (1000, 1499, 2),
    (1500, 1799, 3),
    (2000, 3999, 4),
    (4000, 4999, 5),
    (5000, 5199, 6),
    (5200, 5999, 7),
    (6000, 6799, 8),
    (7000, 8999, 9),
    (9100, 9729, 10),
    (9900, 9999, 11),
)


def sic_division_panel(sic: pd.DataFrame) -> pd.DataFrame:
    """Map a daily SIC-code panel into the project's 11 SIC divisions.

    The returned panel is only a generic ``groups`` input to :func:`score`.
    It has no effect on global winsorization or final gross/net normalization.
    """
    values = sic.to_numpy(dtype=float, na_value=np.nan)
    conditions = [(values >= lower) & (values <= upper) for lower, upper, _ in _SIC_DIVISIONS]
    codes = [code for _, _, code in _SIC_DIVISIONS]
    grouped = np.select(conditions, codes, default=np.nan)
    return pd.DataFrame(grouped, index=sic.index, columns=sic.columns, dtype=float)
