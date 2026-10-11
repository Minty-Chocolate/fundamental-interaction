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
    target = target.copy()
    target.index = pd.to_datetime(target.index)
    daily = daily_return.copy()
    daily.index = pd.to_datetime(daily.index)
    held = target.shift(SIGNAL_TO_RETURN_SESSIONS).reindex(daily.index)
    valid = daily.dropna()
    if valid.empty:
        return pd.DataFrame([{"n_days": 0}])

    wealth = (1.0 + valid).cumprod()
    ann_return = float(wealth.iloc[-1] ** (252.0 / len(valid)) - 1.0) if wealth.iloc[-1] > 0 else float("nan")
    volatility = float(valid.std(ddof=1) * np.sqrt(252.0))
    sharpe = float(valid.mean() / valid.std(ddof=1) * np.sqrt(252.0)) if valid.std(ddof=1) > 0 else float("nan")
    active = held.reindex(valid.index)
    turnover = active.sub(active.shift(1)).abs().sum(axis=1, min_count=1)
    return pd.DataFrame(
        [
            {
                "n_days": len(valid),
                "AnnRet": ann_return,
                "AnnVol": volatility,
                "Sharpe": sharpe,
                "AvgDlyBets": float((active.abs() > nonzero_tolerance).sum(axis=1).mean()),
                "2WayTurnover": float(turnover.mean()),
                "MDD": float((wealth / wealth.cummax() - 1.0).min()),
                "t_stat": _newey_west_t_stat(valid, newey_west_lags),
            }
        ]
    )
