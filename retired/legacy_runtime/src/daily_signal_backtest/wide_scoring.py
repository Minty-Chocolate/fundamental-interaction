"""Cross-sectional signal scoring on date-by-permno wide panels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def global_winsorize(value: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Clip every date's eligible cross-section at its global 1st/99th pct."""
    x = value.where(eligible)
    bounds = x.quantile([0.01, 0.99], axis=1).T
    return x.clip(lower=bounds[0.01], upper=bounds[0.99], axis=0)


def _score_one_group(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return raw z-score, centered percentile-rank, and quintile panels."""
    z = x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1, ddof=0), axis=0)
    rank = x.rank(axis=1, pct=True, method="average")
    quintile = pd.DataFrame(
        np.select([rank.ge(.8), rank.le(.2)], [1.0, -1.0], default=0.0),
        index=x.index,
        columns=x.columns,
    ).where(x.notna())
    return z, rank.sub(.5), quintile


def sic_raw_scores(
    clipped: pd.DataFrame,
    eligible: pd.DataFrame,
    sic_group: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    """Form raw score panels, globally or within same-date SIC divisions.

    ``sic_group`` is a same-shaped panel produced by ``division_panel``.  The
    only loop is over the fixed set of SIC division labels (at most 11); every
    operation within it is vectorized over all dates and securities.
    """
    if sic_group is None:
        z, rank, quintile = _score_one_group(clipped.where(eligible))
        return {"zscore": z, "rank": rank, "quintile": quintile}

    result = {method: pd.DataFrame(np.nan, index=clipped.index, columns=clipped.columns) for method in ("zscore", "rank", "quintile")}
    for division in pd.unique(sic_group.to_numpy().ravel()):
        if pd.isna(division):
            continue
        mask = eligible & sic_group.eq(division)
        z, rank, quintile = _score_one_group(clipped.where(mask))
        result["zscore"] = result["zscore"].where(~mask, z)
        result["rank"] = result["rank"].where(~mask, rank)
        result["quintile"] = result["quintile"].where(~mask, quintile)
    return result


def global_net_gross(raw: pd.DataFrame) -> pd.DataFrame:
    """For each date, enforce net 0 then scale absolute weights to gross 1."""
    centered = raw.sub(raw.mean(axis=1), axis=0)
    return centered.div(centered.abs().sum(axis=1), axis=0)
