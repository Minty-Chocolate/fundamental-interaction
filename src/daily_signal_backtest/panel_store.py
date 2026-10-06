"""Persist compact value panels used by the columnar backtest."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def materialize_value_panels(config: dict) -> None:
    root = Path(config["paths"]["derived_root"])
    output = root / "panels" / "values"
    output.mkdir(parents=True, exist_ok=True)
    signals = json.loads((root / "signals" / "annual_signal_events_manifest.json").read_text())["signals"]
    source = f"read_parquet('{root}/daily_signals/year=*/data.parquet', hive_partitioning=true)"
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA preserve_insertion_order=false")

    panels = [("return", "ret"), ("lagged_market_cap", "lagged_market_cap"), *[(f"signal_{_slug(signal)}", signal) for signal in signals]]
    for name, column in panels:
        destination = output / f"{name}.parquet"
        con.execute(f"COPY (SELECT date, permno, \"{column}\"::DOUBLE AS value FROM {source}) TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        print(f"wrote {destination.name}", flush=True)
    mask = output / "universe_mask.parquet"
    con.execute(
        f"COPY (SELECT date, permno, exchange = 'N' AS is_nyse, NOT (siccd BETWEEN 6000 AND 6999) AS is_nonfinancial, ret IS NOT NULL AS has_return FROM {source}) "
        f"TO '{mask}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    print(f"wrote {mask.name}", flush=True)


def materialize_sic_panel(config: dict) -> None:
    """Write the SIC value panel independently, without rewriting other panels."""
    root = Path(config["paths"]["derived_root"])
    output = root / "panels" / "values"
    output.mkdir(parents=True, exist_ok=True)
    source = f"read_parquet('{root}/daily_signals/year=*/data.parquet', hive_partitioning=true)"
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(
        f"COPY (SELECT date, permno, siccd::INTEGER AS value FROM {source}) "
        f"TO '{output / 'sic.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    print("wrote sic.parquet", flush=True)
