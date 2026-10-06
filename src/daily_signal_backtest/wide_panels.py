"""One-time materialization of global date-by-permno memory-mapped panels."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .panel_store import _slug


def _fill_existing_axis(config: dict, output: Path, fields: dict[str, str]) -> None:
    """Fill selected matrices on an already persisted compact date/permno axis."""
    root = Path(config["paths"]["derived_root"])
    source_root = root / "daily_signals"
    axes = np.load(output / "axes.npz")
    date_axis, permno_axis = axes["dates"], axes["permnos"]
    date_index = {value: idx for idx, value in enumerate(date_axis)}
    permno_index = {int(value): idx for idx, value in enumerate(permno_axis)}
    shape = (len(date_axis), len(permno_axis))
    for name in fields:
        matrix = np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=np.float32, shape=shape)
        matrix[:] = np.nan
        matrix.flush()
        del matrix
    for year_dir in sorted(source_root.glob("year=*")):
        frame = pd.read_parquet(year_dir / "data.parquet", columns=["date", "permno", *fields.values()])
        column_index = frame["permno"].map(permno_index)
        frame = frame.loc[column_index.notna()].copy()
        column_index = column_index.loc[frame.index]
        row = np.fromiter((date_index[value] for value in pd.to_datetime(frame["date"]).to_numpy(dtype="datetime64[ns]")), dtype=np.int64, count=len(frame))
        col = column_index.to_numpy(dtype=np.int64)
        for name, column in fields.items():
            matrix = np.load(output / f"{name}.npy", mmap_mode="r+")
            matrix[row, col] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan)
            matrix.flush()
            del matrix
        print(f"materialized {year_dir.name}", flush=True)


def materialize_base_wide_panels(config: dict) -> None:
    """Build only the three panels required by the return engine immediately."""
    root = Path(config["paths"]["derived_root"])
    output = Path(config["engine"].get("wide_cache_root", root / "wide"))
    if not (output / "axes.npz").exists():
        materialize_wide_panels(config)
        return
    _fill_existing_axis(config, output, {"return": "ret", "lagged_market_cap": "lagged_market_cap", "sic": "siccd"})
    print("wrote 3 base wide panels", flush=True)


def materialize_signal_wide_panels(config: dict) -> None:
    """Persist the 11 canonical signal panels after the return engine is ready."""
    root = Path(config["paths"]["derived_root"])
    output = Path(config["engine"].get("wide_cache_root", root / "wide"))
    signals = json.loads((root / "signals" / "annual_signal_events_manifest.json").read_text())["signals"]
    _fill_existing_axis(config, output, {f"signal_{_slug(signal)}": signal for signal in signals})
    print(f"wrote {len(signals)} signal wide panels", flush=True)


def materialize_wide_panels(config: dict) -> None:
    """Write return/cap/SIC/signals as dense float32 ``.npy`` matrices.

    Reading the construction panel year-by-year limits build memory.  The
    resulting panels share one permanent full-sample date and permno axis and
    can subsequently be opened with ``np.load(..., mmap_mode='r')``.
    """
    root = Path(config["paths"]["derived_root"])
    source_root = root / "daily_signals"
    # Wide panels are cache artifacts, not source data. Keep them on the local
    # SSD and make their axis the largest universe ever needed by the grid.
    output = Path(config["engine"].get("wide_cache_root", root / "wide"))
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    signals = json.loads((root / "signals" / "annual_signal_events_manifest.json").read_text())["signals"]
    years = sorted(source_root.glob("year=*"))

    dates, permnos = set(), set()
    minimum_cutoff = min(config["grid"]["microcap_percentiles"])
    for year_dir in years:
        frame = pd.read_parquet(year_dir / "data.parquet", columns=["date", "permno", "ret", "lagged_market_cap", "siccd"])
        dates.update(pd.to_datetime(frame["date"]).unique())
        base = frame["ret"].notna() & frame["lagged_market_cap"].gt(0) & ~frame["siccd"].between(6000, 6999)
        cutoffs = frame.loc[base].groupby("date")["lagged_market_cap"].quantile(minimum_cutoff)
        threshold = pd.to_datetime(frame["date"]).map(cutoffs)
        permnos.update(frame.loc[base & frame["lagged_market_cap"].ge(threshold), "permno"].astype("int64").unique())
    date_axis = np.array(sorted(dates), dtype="datetime64[ns]")
    permno_axis = np.array(sorted(permnos), dtype=np.int64)
    np.savez_compressed(output / "axes.npz", dates=date_axis, permnos=permno_axis)
    date_index = {value: idx for idx, value in enumerate(date_axis)}
    permno_index = {int(value): idx for idx, value in enumerate(permno_axis)}

    fields = {"return": "ret", "lagged_market_cap": "lagged_market_cap", "sic": "siccd"}
    fields.update({f"signal_{_slug(signal)}": signal for signal in signals})
    shape = (len(date_axis), len(permno_axis))
    for name in fields:
        matrix = np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=np.float32, shape=shape)
        matrix[:] = np.nan
        del matrix

    for year_dir in years:
        frame = pd.read_parquet(year_dir / "data.parquet", columns=["date", "permno", *fields.values()])
        # The compact axis intentionally omits names that never pass the
        # minimum cutoff. Do not reintroduce those rows while filling values.
        column_index = frame["permno"].map(permno_index)
        frame = frame.loc[column_index.notna()].copy()
        column_index = column_index.loc[frame.index]
        row = np.fromiter((date_index[value] for value in pd.to_datetime(frame["date"]).to_numpy(dtype="datetime64[ns]")), dtype=np.int64, count=len(frame))
        col = column_index.to_numpy(dtype=np.int64)
        for name, column in fields.items():
            matrix = np.load(output / f"{name}.npy", mmap_mode="r+")
            matrix[row, col] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan)
            del matrix
        print(f"materialized {year_dir.name}", flush=True)
    print(f"wrote {len(fields)} compact wide panels with shape={shape}; excluded names never above {minimum_cutoff:.0%} cutoff", flush=True)
