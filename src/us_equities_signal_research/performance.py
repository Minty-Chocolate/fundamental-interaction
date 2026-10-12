"""Portfolio metrics for daily long-short returns under the fixed timing rule."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .execution import SIGNAL_TO_RETURN_SESSIONS
from .model import ScoredPanel


def _newey_west_t_stat(values: pd.Series, max_lag: int = 5) -> float:
    x = values.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    demeaned = x - x.mean()
    lrv = float(np.dot(demeaned, demeaned) / n)
    for lag in range(1, min(max_lag, n - 1) + 1):
        covariance = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        lrv += 2.0 * (1.0 - lag / (max_lag + 1.0)) * covariance
    if lrv <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(lrv / n))


def performance_metrics(
    daily_return: pd.Series,
    weights: ScoredPanel | pd.DataFrame,
    *,
    newey_west_lags: int = 5,
    nonzero_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Return one review-ready metrics row for an already formed portfolio.

    Bets and turnover are measured on held weights (the same two-session shift
    used for returns), not on unexecuted after-close targets.
    """
    target = weights.values if isinstance(weights, ScoredPanel) else weights
    daily = daily_return.copy()
    daily.index = pd.to_datetime(daily.index)
    valid = daily.dropna()
    if valid.empty:
        return pd.DataFrame([{"n_days": 0}])

    target_index = pd.DatetimeIndex(target.index.to_numpy(dtype="datetime64[ns]"))
    valid_index = pd.DatetimeIndex(valid.index.to_numpy(dtype="datetime64[ns]"))
    locations = target_index.get_indexer(valid_index) - SIGNAL_TO_RETURN_SESSIONS
    if (locations < 0).any():
        raise ValueError("Daily returns precede the available held-weight state.")
    values = target.to_numpy(dtype=float, copy=False)
    # Resolved research universes use two opening state sessions, so this is a
    # view instead of ``shift`` plus ``reindex`` and their two full copies.
    if np.array_equal(locations, np.arange(locations[0], locations[0] + len(locations))):
        active = values[locations[0] : locations[-1] + 1]
    else:
        active = values[locations]

    wealth = (1.0 + valid).cumprod()
    ann_return = float(wealth.iloc[-1] ** (252.0 / len(valid)) - 1.0) if wealth.iloc[-1] > 0 else float("nan")
    volatility = float(valid.std(ddof=1) * np.sqrt(252.0))
    sharpe = float(valid.mean() / valid.std(ddof=1) * np.sqrt(252.0)) if valid.std(ddof=1) > 0 else float("nan")
    bets = float((np.abs(active) > nonzero_tolerance).sum(axis=1).mean())
    turnover_total = 0.0
    turnover_days = 0
    previous = active[0]
    # Avoid allocating a date × permno difference frame for every grid point.
    for start in range(1, len(active), 256):
        current = active[start : start + 256]
        prior = np.empty_like(current)
        prior[0] = previous
        prior[1:] = current[:-1]
        delta = np.abs(current - prior)
        finite = np.isfinite(delta)
        row_total = np.where(finite, delta, 0.0).sum(axis=1)
        row_used = finite.any(axis=1)
        turnover_total += float(row_total[row_used].sum())
        turnover_days += int(row_used.sum())
        previous = current[-1]
    return pd.DataFrame(
        [
            {
                "n_days": len(valid),
                "AnnRet": ann_return,
                "AnnVol": volatility,
                "Sharpe": sharpe,
                "AvgDlyBets": bets,
                "2WayTurnover": float(turnover_total / turnover_days) if turnover_days else float("nan"),
                "MDD": float((wealth / wealth.cummax() - 1.0).min()),
                "t_stat": _newey_west_t_stat(valid, newey_west_lags),
            }
        ]
    )
