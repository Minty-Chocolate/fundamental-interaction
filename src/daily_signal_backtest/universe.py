from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd


CRSP_COLUMNS = [
    "PERMNO", "DlyCalDt", "DlyRet", "DlyCap", "DlyPrevCap", "PrimaryExch",
    "SecurityType", "SecuritySubType", "ShareType", "TradingStatusFlg", "SICCD", "DlyDelFlg",
]


def extract_daily_stock_base(config: dict) -> None:
    """Stream CIZ daily CRSP once and create a compact, immutable base table.

    No microcap or signal screen occurs here.  DlyPrevCap is retained solely
    for later NYSE breakpoint construction, keeping universe rules decoupled.
    """
    root = Path(config["paths"]["derived_root"]) / "daily_stock_base"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"{manifest_path} exists; remove it only to intentionally rebuild the base table")
    source = config["paths"]["crsp_daily_zip"]
    rows_written = 0
    parts = 0
    with zipfile.ZipFile(source) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(csv_name) as handle:
            for number, chunk in enumerate(pd.read_csv(handle, usecols=CRSP_COLUMNS, chunksize=500_000, low_memory=False)):
                chunk["DlyCalDt"] = pd.to_datetime(chunk["DlyCalDt"], errors="coerce")
                chunk["DlyRet"] = pd.to_numeric(chunk["DlyRet"], errors="coerce")
                chunk["DlyCap"] = pd.to_numeric(chunk["DlyCap"], errors="coerce")
                chunk["DlyPrevCap"] = pd.to_numeric(chunk["DlyPrevCap"], errors="coerce")
                # CIZ equivalent of the paper's CRSP common-share and
                # NYSE/AMEX/NASDAQ universe. Financial exclusion is deferred.
                keep = (
                    chunk["DlyCalDt"].between("1999-01-01", config["sample"]["end_date"])
                    & chunk["SecurityType"].eq("EQTY")
                    & chunk["SecuritySubType"].eq("COM")
                    & chunk["PrimaryExch"].isin(["N", "A", "Q"])
                )
                frame = chunk.loc[keep, CRSP_COLUMNS].rename(columns={
                    "PERMNO": "permno", "DlyCalDt": "date", "DlyRet": "ret",
                    "DlyCap": "market_cap", "DlyPrevCap": "lagged_market_cap",
                    "PrimaryExch": "exchange", "SICCD": "siccd", "DlyDelFlg": "delisting_flag",
                })
                if frame.empty:
                    continue
                frame.to_parquet(root / f"part-{number:05d}.parquet", index=False, compression="zstd")
                rows_written += len(frame)
                parts += 1
                if parts % 20 == 0:
                    print(f"wrote {parts} parts / {rows_written:,} rows", flush=True)
    manifest = {
        "rows_written": rows_written,
        "parts": parts,
        "date_range_read": ["1999-01-01", config["sample"]["end_date"]],
        "security_filter": {"SecurityType": "EQTY", "SecuritySubType": "COM", "PrimaryExch": ["N", "A", "Q"]},
        "microcap_not_applied": True,
        "financials_not_applied": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
