"""One-time daily NYSE membership matrix for the wide-panel backtest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def materialize_nyse_panel(config: dict) -> None:
    root = Path(config["paths"]["derived_root"])
    source_root, output = root / "daily_signals", root / "wide"
    axes = np.load(output / "axes.npz")
    dates, permnos = axes["dates"], axes["permnos"]
    date_index = {value: idx for idx, value in enumerate(dates)}
    permno_index = {int(value): idx for idx, value in enumerate(permnos)}
    panel = np.lib.format.open_memmap(output / "is_nyse.npy", mode="w+", dtype=np.bool_, shape=(len(dates), len(permnos)))
    panel[:] = False
    for year_dir in sorted(source_root.glob("year=*")):
        frame = pd.read_parquet(year_dir / "data.parquet", columns=["date", "permno", "exchange"])
        row = np.fromiter((date_index[value] for value in pd.to_datetime(frame["date"]).to_numpy(dtype="datetime64[ns]")), dtype=np.int64, count=len(frame))
        col = frame["permno"].map(permno_index).to_numpy(dtype=np.int64)
        panel[row, col] = frame["exchange"].eq("N").to_numpy()
        print(f"materialized NYSE {year_dir.name}", flush=True)
    panel.flush()
