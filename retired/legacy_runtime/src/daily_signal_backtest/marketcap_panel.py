from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_marketcap_panels(config: dict) -> None:
    """Write annual date-by-PERMNO market-cap panels and row-level masks."""
    root = Path(config["paths"]["derived_root"])
    output = root / "panels" / "market_cap"
    output.mkdir(parents=True, exist_ok=True)
    columns = ["date", "permno", "lagged_market_cap", "exchange", "siccd", "ret"]
    for year in range(2000, 2026):
        frame = pd.read_parquet(root / "daily_signals" / f"year={year}" / "data.parquet", columns=columns)
        frame["permno"] = frame["permno"].astype("int32")
        frame["lagged_market_cap"] = frame["lagged_market_cap"].astype("float32")
        panel = frame.pivot(index="date", columns="permno", values="lagged_market_cap")
        panel.columns = panel.columns.astype(str)
        panel.to_parquet(output / f"year={year}.parquet", compression="zstd")
        masks = frame.assign(
            is_nyse=frame["exchange"].eq("N"),
            is_nonfinancial=~frame["siccd"].between(config["sample"]["financial_sic_min"], config["sample"]["financial_sic_max"]),
            has_return=frame["ret"].notna(),
        )[["date", "permno", "is_nyse", "is_nonfinancial", "has_return"]]
        masks.to_parquet(output / f"universe_year={year}.parquet", index=False, compression="zstd")
        print(f"market-cap panel {year}: {panel.shape[0]} dates x {panel.shape[1]} stocks", flush=True)
