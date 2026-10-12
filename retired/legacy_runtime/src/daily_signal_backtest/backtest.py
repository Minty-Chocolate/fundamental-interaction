"""Portfolio aggregation entry points.

Legacy per-date pandas loops were intentionally removed.  Weight formation now
lives in :mod:`daily_signal_backtest.vectorized_weights` and is panel-level.
"""

from __future__ import annotations


def run_backtest(config: dict) -> None:
    raise RuntimeError(
        "The legacy daily-loop backtest was removed. "
        "Use the vectorized portfolio aggregation runner."
    )


def run_backtest_grid(config: dict) -> None:
    raise RuntimeError(
        "The legacy grid loop was removed. "
        "Use the vectorized portfolio aggregation runner."
    )
