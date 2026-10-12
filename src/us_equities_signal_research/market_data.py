"""Cached CRSP wide panels used alongside cached raw signal panels."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .cache import cache_root, universe_key


def _path(field: str, universe: object) -> Path:
    return cache_root() / "crsp" / field / f"{universe_key(universe)}.parquet"


def panel(field: str, *, data: object, universe: object, refresh: bool = False) -> pd.DataFrame:
    """Return a cached CRSP daily wide panel for a resolved dynamic universe.

    ``ret``, ``market_cap``, and ``siccd`` are normal calls.  Values are
    membership-masked by DataHandler; the period-wide union only defines the
    output columns and never changes the economic membership mask.
    """
    resolved = universe if hasattr(universe, "membership") else data.universe.resolve(universe, opening_state_sessions=2)
    path = _path(field, resolved)
    manifest = path.with_suffix(".json")
    if not refresh and path.is_file() and manifest.is_file():
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index)
        frame.columns = pd.Index([int(column) for column in frame.columns], dtype="int64")
        return frame

    frame = data.crsp.panel(field, universe=resolved).to_pandas().set_index("date")
    frame.index = pd.to_datetime(frame.index)
    frame.columns = pd.Index([int(column) for column in frame.columns], dtype="int64")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.parquet")
    frame.to_parquet(temporary)
    temporary.replace(path)
    manifest.write_text(
        json.dumps(
            {
                "field": field,
                "universe_key": universe_key(resolved),
                "state_start": str(resolved.state_start),
                "report_start": str(resolved.report_start),
                "n_dates": len(frame.index),
                "n_permnos": len(frame.columns),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return frame
