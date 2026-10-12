"""Public immutable containers for the standalone PIT research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
import polars as pl


@dataclass(frozen=True)
class FundamentalEvents:
    """Universe-scoped, GVKEY-keyed source events supplied to one recipe."""

    annual: pl.DataFrame
    data: object
    coverage_permnos: tuple[int, ...]
    annual_pandas: pd.DataFrame | None = None


class Recipe(Protocol):
    """A signal definition with exactly one executable method."""

    name: str
    required_fields: tuple[str, ...]
    annual_lookback_years: int
    prefetch_buffer_years: int
    version: str

    def compute(self, events: FundamentalEvents, calendar: pd.DatetimeIndex) -> pd.DataFrame:
        """Return a raw daily ``date × permno`` signal panel."""


@dataclass(frozen=True)
class RawSignalPanel:
    name: str
    values: pd.DataFrame
    universe: object
    cache_path: Path | None = None


@dataclass(frozen=True)
class ScoredPanel:
    method: str
    values: pd.DataFrame
    universe: object
    normalized: bool
