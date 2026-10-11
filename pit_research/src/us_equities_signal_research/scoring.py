"""Cross-sectional scoring of daily raw panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import RawSignalPanel, ScoredPanel


_METHODS = frozenset({"zscore", "rank", "quintile"})
_NORMALIZATIONS = frozenset({None, "net_zero_gross_one"})


def _membership_frame(universe: object, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    membership = universe.membership.to_pandas().set_index("date")
    membership.index = pd.to_datetime(membership.index)
    membership.columns = pd.Index([int(column) for column in membership.columns], dtype="int64")
    return membership.reindex(index=index, columns=columns, fill_value=False).fillna(False).astype(bool)


def _coerce_groups(groups: pd.DataFrame | None, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame | None:
    if groups is None:
        return None
    frame = groups.copy()
    frame.index = pd.to_datetime(frame.index)
    frame.columns = pd.Index([int(column) for column in frame.columns], dtype="int64")
    return frame.reindex(index=index, columns=columns)


def _score_cross_section(values: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "zscore":
        deviation = values.std(axis=1, ddof=0).replace(0.0, np.nan)
        zscore = values.sub(values.mean(axis=1), axis=0).div(deviation, axis=0)
        # A finite but tied eligible set is a valid zero signal, not missing
        # information.  Preserve missing inputs as missing.
        return zscore.where(deviation.notna(), values.where(values.isna(), 0.0))
    percentile = values.rank(axis=1, pct=True, method="average")
    if method == "rank":
        return percentile.mul(2.0).sub(1.0)
    return pd.DataFrame(
        np.select([percentile.le(0.2), percentile.ge(0.8)], [-1.0, 1.0], default=0.0),
        index=values.index,
        columns=values.columns,
    ).where(values.notna())


def _global_net_zero_gross_one(values: pd.DataFrame) -> pd.DataFrame:
    centered = values.sub(values.mean(axis=1), axis=0)
    gross = centered.abs().sum(axis=1)
    weights = centered.div(gross.replace(0.0, np.nan), axis=0)
    # A valid but completely tied cross-section has zero exposure, whereas a
    # missing signal remains missing.  Preserve that distinction explicitly.
    tied = gross.eq(0.0)
    if tied.any():
        weights.loc[tied] = values.loc[tied].where(values.loc[tied].isna(), 0.0)
    return weights


def score(
    raw_signal: RawSignalPanel | pd.DataFrame,
    *,
    universe: object | None = None,
    groups: pd.DataFrame | None = None,
    method: str = "zscore",
    winsorize: float = 0.01,
    normalize: str | None = "net_zero_gross_one",
) -> ScoredPanel:
    """Transform a raw daily panel into scores or portfolio weights.

    The eligible observation set is exactly ``membership[t] & signal[t].notna``.
    Winsor bounds are always global per date; group labels only affect the
    requested z-score/rank/quintile transformation.  Normalization is global,
    so sector membership never creates sector-level gross or net constraints.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}")
    if normalize not in _NORMALIZATIONS:
        raise ValueError("normalize must be None or 'net_zero_gross_one'.")
    if not 0.0 <= winsorize < 0.5:
        raise ValueError("winsorize must be in [0, 0.5).")

    if isinstance(raw_signal, RawSignalPanel):
        values = raw_signal.values.copy()
        resolved = raw_signal.universe
        if universe is not None and getattr(universe, "rule", universe) != resolved.rule:
            raise ValueError("The supplied universe does not match the raw signal panel.")
    else:
        if universe is None or not hasattr(universe, "membership"):
            raise ValueError("A resolved universe is required when raw_signal is a DataFrame.")
        values, resolved = raw_signal.copy(), universe
    values.index = pd.to_datetime(values.index)
    values.columns = pd.Index([int(column) for column in values.columns], dtype="int64")

    membership = _membership_frame(resolved, values.index, values.columns)
    eligible = membership & values.notna()
    masked = values.where(eligible)
    if winsorize:
        bounds = masked.quantile([winsorize, 1.0 - winsorize], axis=1).T
        clipped = masked.clip(lower=bounds[winsorize], upper=bounds[1.0 - winsorize], axis=0)
    else:
        clipped = masked

    group_frame = _coerce_groups(groups, values.index, values.columns)
    if group_frame is None:
        transformed = _score_cross_section(clipped, method)
    else:
        transformed = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
        for label in pd.unique(group_frame.to_numpy().ravel()):
            if pd.isna(label):
                continue
            group_mask = eligible & group_frame.eq(label)
            local = _score_cross_section(clipped.where(group_mask), method)
            transformed = transformed.where(~group_mask, local)

    result = _global_net_zero_gross_one(transformed) if normalize else transformed
    return ScoredPanel(method=method, values=result.where(eligible), universe=resolved, normalized=normalize is not None)
