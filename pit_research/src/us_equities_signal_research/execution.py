"""Realised PnL under the fixed after-close signal timing convention."""

from __future__ import annotations

import pandas as pd

from .model import ScoredPanel


SIGNAL_TO_RETURN_SESSIONS = 2


def realized_returns(weights: ScoredPanel | pd.DataFrame, returns: pd.DataFrame, *, report_start: object | None = None) -> pd.Series:
    """Return daily long-short PnL using ``target[t-2] * ret[t]``.

    A signal observed after close on ``t`` produces a target at ``t``; it forms
    the position at the ``t+1`` close and earns the ``t+1``→``t+2`` return.
    Therefore target weights are shifted by exactly two CRSP trading sessions.
    """
    target = weights.values if isinstance(weights, ScoredPanel) else weights
    target = target.copy()
    target.index = pd.to_datetime(target.index)
    target.columns = pd.Index([int(column) for column in target.columns], dtype="int64")
    ret = returns.copy()
    ret.index = pd.to_datetime(ret.index)
    ret.columns = pd.Index([int(column) for column in ret.columns], dtype="int64")
    ret = ret.reindex(index=target.index, columns=target.columns)
    pnl = target.shift(SIGNAL_TO_RETURN_SESSIONS).mul(ret).sum(axis=1, min_count=1)
    start = report_start
    if start is None and isinstance(weights, ScoredPanel):
        start = weights.universe.report_start
    return pnl if start is None else pnl.loc[pnl.index >= pd.Timestamp(start)]
