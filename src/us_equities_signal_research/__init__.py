"""Independent point-in-time daily-signal research API."""

from .execution import SIGNAL_TO_RETURN_SESSIONS, realized_returns
from .groups import sic_division_panel
from .market_data import panel as crsp_panel
from .model import RawSignalPanel, Recipe, ScoredPanel
from .performance import performance_metrics
from .pipeline import panel
from .scoring import score
from .recipes import get_recipe, signal_names
from .backtest import excess_returns, rebalance_targets, run as run_backtest
from .pipeline import panels

__all__ = [
    "RawSignalPanel",
    "Recipe",
    "SIGNAL_TO_RETURN_SESSIONS",
    "crsp_panel",
    "ScoredPanel",
    "panel",
    "performance_metrics",
    "realized_returns",
    "score",
    "sic_division_panel",
    "get_recipe",
    "signal_names",
    "panels",
    "excess_returns",
    "rebalance_targets",
    "run_backtest",
]
