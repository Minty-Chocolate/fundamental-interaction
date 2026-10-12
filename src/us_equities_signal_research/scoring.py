"""Cross-sectional scoring of daily raw panels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import RawSignalPanel, ScoredPanel


_METHODS = frozenset({"zscore", "rank", "quintile"})
_NORMALIZATIONS = frozenset({None, "net_zero_gross_one"})


@dataclass(frozen=True)
class PreparedScore:
    """Reusable, globally winsorised score inputs for one raw signal panel."""

    clipped: pd.DataFrame
    eligible: pd.DataFrame
    universe: object


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


def _score_cross_sections(
    values: pd.DataFrame, methods: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Calculate requested global transforms, sharing their rank pass."""
    result: dict[str, pd.DataFrame] = {}
    if "zscore" in methods:
        deviation = values.std(axis=1, ddof=0).replace(0.0, np.nan)
        zscore = values.sub(values.mean(axis=1), axis=0).div(deviation, axis=0)
        result["zscore"] = zscore.where(deviation.notna(), values.where(values.isna(), 0.0))
    if {"rank", "quintile"}.intersection(methods):
        percentile = values.rank(axis=1, pct=True, method="average")
        if "rank" in methods:
            result["rank"] = percentile.mul(2.0).sub(1.0)
        if "quintile" in methods:
            result["quintile"] = pd.DataFrame(
                np.select([percentile.le(0.2), percentile.ge(0.8)], [-1.0, 1.0], default=0.0),
                index=values.index,
                columns=values.columns,
            ).where(values.notna())
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based average ranks, matching ``Series.rank(method='average')``."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
    ends = np.r_[starts[1:] - 1, len(values) - 1]
    sorted_ranks = np.repeat((starts + ends + 2.0) / 2.0, ends - starts + 1)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def _grouped_scores(
    values: pd.DataFrame,
    eligible: pd.DataFrame,
    groups: pd.DataFrame,
    methods: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Score all requested methods in one pass over date × SIC subgroups.

    The former implementation made a whole-panel ``where`` and rank pass for
    every SIC division and method.  This implementation visits only eligible
    firms within each date, calculates every requested transform there, then
    moves to the next date.
    """
    value_array = values.to_numpy(dtype=float, copy=False)
    eligible_array = eligible.to_numpy(dtype=bool, copy=False)
    try:
        group_array = groups.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError):
        label_array = groups.to_numpy(dtype=object, na_value=None)
        group_array = np.full(label_array.shape, np.nan)
        for row, labels in enumerate(label_array):
            valid = pd.notna(labels)
            group_array[row, valid] = pd.factorize(labels[valid], sort=False)[0]

    output = {method: np.full(value_array.shape, np.nan, dtype=float) for method in methods}
    for row in range(value_array.shape[0]):
        valid = eligible_array[row] & np.isfinite(group_array[row])
        positions = np.flatnonzero(valid)
        if not len(positions):
            continue
        row_groups = group_array[row, positions]
        group_order = np.argsort(row_groups, kind="mergesort")
        positions = positions[group_order]
        ordered_groups = row_groups[group_order]
        starts = np.r_[0, np.flatnonzero(ordered_groups[1:] != ordered_groups[:-1]) + 1]
        ends = np.r_[starts[1:], len(positions)]
        for start, end in zip(starts, ends, strict=True):
            position = positions[start:end]
            local = value_array[row, position]
            count = len(local)
            if "zscore" in output:
                deviation = local.std(ddof=0)
                output["zscore"][row, position] = 0.0 if deviation == 0.0 else (local - local.mean()) / deviation
            if {"rank", "quintile"}.intersection(output):
                percentile = _average_ranks(local) / count
                if "rank" in output:
                    output["rank"][row, position] = percentile * 2.0 - 1.0
                if "quintile" in output:
                    output["quintile"][row, position] = np.select(
                        [percentile <= 0.2, percentile >= 0.8], [-1.0, 1.0], default=0.0
                    )
    return {
        method: pd.DataFrame(array, index=values.index, columns=values.columns)
        for method, array in output.items()
    }


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


def prepare_score(
    raw_signal: RawSignalPanel | pd.DataFrame,
    *,
    universe: object | None = None,
    winsorize: float = 0.01,
) -> PreparedScore:
    """Apply the shared eligibility mask and global winsorisation once."""
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
    clipped = values.where(eligible)
    if winsorize:
        bounds = clipped.quantile([winsorize, 1.0 - winsorize], axis=1).T
        clipped = clipped.clip(lower=bounds[winsorize], upper=bounds[1.0 - winsorize], axis=0)
    return PreparedScore(clipped=clipped, eligible=eligible, universe=resolved)


def score_prepared(
    prepared: PreparedScore,
    *,
    groups: pd.DataFrame | None = None,
    method: str = "zscore",
    normalize: str | None = "net_zero_gross_one",
) -> ScoredPanel:
    """Score prepared inputs under one method and optional group panel."""
    return score_prepared_many(
        prepared, groups=groups, methods=(method,), normalize=normalize,
    )[method]


def score_prepared_many(
    prepared: PreparedScore,
    *,
    groups: pd.DataFrame | None = None,
    methods: tuple[str, ...] = ("zscore", "rank", "quintile"),
    normalize: str | None = "net_zero_gross_one",
) -> dict[str, ScoredPanel]:
    """Score several methods while sharing their expensive preprocessing."""
    methods = tuple(dict.fromkeys(methods))
    if not methods or any(method not in _METHODS for method in methods):
        raise ValueError(f"methods must be a non-empty subset of {sorted(_METHODS)}")
    if normalize not in _NORMALIZATIONS:
        raise ValueError("normalize must be None or 'net_zero_gross_one'.")

    clipped, eligible = prepared.clipped, prepared.eligible
    group_frame = _coerce_groups(groups, clipped.index, clipped.columns)
    if group_frame is None:
        transformed_by_method = _score_cross_sections(clipped, methods)
    else:
        transformed_by_method = _grouped_scores(clipped, eligible, group_frame, methods)

    return {
        method: ScoredPanel(
            method=method,
            values=(_global_net_zero_gross_one(transformed) if normalize else transformed).where(eligible),
            universe=prepared.universe,
            normalized=normalize is not None,
        )
        for method, transformed in transformed_by_method.items()
    }


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
    prepared = prepare_score(raw_signal, universe=universe, winsorize=winsorize)
    return score_prepared(prepared, groups=groups, method=method, normalize=normalize)
