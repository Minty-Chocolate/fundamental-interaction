"""Small, explicit cache for daily wide panels on the fast local volume."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def cache_root() -> Path:
    return Path(os.environ.get("US_EQUITIES_SIGNAL_CACHE", "data/derived/pit_signal_panels"))


def universe_key(universe: object) -> str:
    rule = universe.rule
    criterion = (
        f"rank{rule.rank_lte}" if rule.rank_lte is not None else
        f"quantile{rule.quantile_lte:g}" if rule.quantile_lte is not None else
        f"cumpct{rule.cumpct_lte:g}"
    )
    return f"{criterion}_lag{rule.lag}_{rule.start:%Y%m%d}_{rule.end:%Y%m%d}"


def raw_path(name: str, universe: object, version: str) -> Path:
    return cache_root() / "raw" / name / version / f"{universe_key(universe)}.parquet"


def load_raw(name: str, universe: object, version: str) -> tuple[pd.DataFrame, Path] | None:
    path = raw_path(name, universe, version)
    manifest = path.with_suffix(".json")
    if not path.is_file() or not manifest.is_file():
        return None
    metadata = json.loads(manifest.read_text())
    if metadata.get("name") != name or metadata.get("universe_key") != universe_key(universe):
        return None
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index)
    frame.columns = pd.Index([int(column) for column in frame.columns], dtype="int64")
    return frame, path


def store_raw(name: str, universe: object, version: str, values: pd.DataFrame, metadata: dict) -> Path:
    path = raw_path(name, universe, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.parquet")
    values.to_parquet(temporary)
    temporary.replace(path)
    manifest = {
        "name": name,
        "universe_key": universe_key(universe),
        "version": version,
        "state_start": str(universe.state_start),
        "report_start": str(universe.report_start),
        "end": str(universe.end),
        "n_dates": len(values.index),
        "n_permnos": len(values.columns),
        **metadata,
    }
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path
