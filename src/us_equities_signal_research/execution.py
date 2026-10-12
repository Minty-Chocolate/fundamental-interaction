"""Realised PnL under the fixed after-close signal timing convention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import ScoredPanel


SIGNAL_TO_RETURN_SESSIONS = 2


@dataclass(frozen=True)
class ReturnObservations:
    """Sparse index of the non-missing values in a fixed return panel.

    CRSP panels remain wide and cacheable.  A dynamic top-N universe leaves
    most period-wide union columns missing on any one day, so this index lets
    each portfolio PnL use only securities with an observable daily return.
    """

    index: pd.DatetimeIndex
    columns: pd.Index
    row: np.ndarray
    column: np.ndarray


class DenseReturnEngine:
    """Fixed excess-return matrix reused across one signal's full grid.

    Dynamic top-N panels are sparse, but repeatedly gathering their sparse
    coordinates became slower than a contiguous NumPy contraction for every
    frequency × score combination.  The engine materialises the zero-filled
    adjusted-return matrix once, then only allocates the current target slice.
    """

    def __init__(self, returns: pd.DataFrame, *, report_start: object | None = None) -> None:
        self.index = pd.DatetimeIndex(returns.index.to_numpy(dtype="datetime64[ns]"))
        raw = returns.to_numpy(dtype=float, copy=False)
        self.values = np.nan_to_num(raw, nan=0.0)
        self.observable = np.isfinite(raw)
        self.report_start = None if report_start is None else pd.Timestamp(report_start)

    def realized(self, weights: ScoredPanel | pd.DataFrame) -> pd.Series:
        target = weights.values if isinstance(weights, ScoredPanel) else weights
        index = pd.DatetimeIndex(target.index.to_numpy(dtype="datetime64[ns]"))
        if not index.equals(self.index):
            raise ValueError("Target and return indices must match.")
        matrix = target.to_numpy(dtype=float, copy=False)
        pnl_values = np.full(len(index), np.nan, dtype=float)
        lagged = matrix[:-SIGNAL_TO_RETURN_SESSIONS]
        pnl_values[SIGNAL_TO_RETURN_SESSIONS:] = np.einsum(
            "ij,ij->i",
            np.nan_to_num(lagged, nan=0.0),
            self.values[SIGNAL_TO_RETURN_SESSIONS:],
        )
        usable = (np.isfinite(lagged) & self.observable[SIGNAL_TO_RETURN_SESSIONS:]).any(axis=1)
        pnl_values[SIGNAL_TO_RETURN_SESSIONS:][~usable] = np.nan
        pnl = pd.Series(pnl_values, index=index, dtype=float)
        return pnl if self.report_start is None else pnl.loc[pnl.index >= self.report_start]


def return_observations(returns: pd.DataFrame) -> ReturnObservations:
    """Build the reusable sparse observation index for a wide return panel."""
    index = pd.DatetimeIndex(pd.to_datetime(returns.index))
    values = returns.to_numpy(dtype=float, copy=False)
    row, column = np.nonzero(np.isfinite(values))
    return ReturnObservations(index=index, columns=returns.columns.copy(), row=row, column=column)


def realized_returns(
    weights: ScoredPanel | pd.DataFrame,
    returns: pd.DataFrame,
    *,
    report_start: object | None = None,
    observations: ReturnObservations | None = None,
) -> pd.Series:
    """Return daily long-short PnL using ``target[t-2] * ret[t]``.

    A signal observed after close on ``t`` produces a target at ``t``; it forms
    the position at the ``t+1`` close and earns the ``t+1``→``t+2`` return.
    Therefore target weights are shifted by exactly two CRSP trading sessions.
    With ``observations`` supplied, the calculation visits only non-missing
    CRSP returns and has the same ``min_count=1`` behavior as the dense form.
    """
    target = weights.values if isinstance(weights, ScoredPanel) else weights
    target = target.copy()
    target.index = pd.DatetimeIndex(pd.to_datetime(target.index))
    target.columns = pd.Index([int(column) for column in target.columns], dtype="int64")
    ret = returns.copy()
    ret.index = pd.DatetimeIndex(pd.to_datetime(ret.index))
    ret.columns = pd.Index([int(column) for column in ret.columns], dtype="int64")
    ret = ret.reindex(index=target.index, columns=target.columns)

    if (
        observations is None
        or not observations.index.equals(target.index)
        or not observations.columns.equals(target.columns)
    ):
        observations = return_observations(ret)

    matrix = target.to_numpy(dtype=float, copy=False)
    usable = observations.row >= SIGNAL_TO_RETURN_SESSIONS
    row = observations.row[usable]
    column = observations.column[usable]
    current_returns = ret.to_numpy(dtype=float, copy=False)[row, column]
    lagged_weights = matrix[row - SIGNAL_TO_RETURN_SESSIONS, column]
    valid = np.isfinite(lagged_weights) & np.isfinite(current_returns)

    pnl_values = np.full(len(target.index), np.nan, dtype=float)
    if valid.any():
        observed_rows = row[valid]
        contribution = lagged_weights[valid] * current_returns[valid]
        total = np.bincount(observed_rows, weights=contribution, minlength=len(target.index))
        count = np.bincount(observed_rows, minlength=len(target.index))
        pnl_values[count > 0] = total[count > 0]
    pnl = pd.Series(pnl_values, index=target.index, dtype=float)

    start = report_start
    if start is None and isinstance(weights, ScoredPanel):
        start = weights.universe.report_start
    return pnl if start is None else pnl.loc[pnl.index >= pd.Timestamp(start)]
