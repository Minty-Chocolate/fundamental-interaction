from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd


ANNUAL_COLUMNS = [
    "gvkey", "datadate", "fyear", "fyr", "indfmt", "datafmt", "consol",
    "pdate", "fdate", "final",
]


def audit_annual(config: dict) -> None:
    """Audit availability dates and repeated Annual observations.

    PDATE is audit-only.  The production availability convention is FDATE,
    falling back to DATADATE plus six calendar months when FDATE is missing.
    """
    out = Path(config["paths"]["derived_root"]) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    path = config["paths"]["compustat_annual_zip"]
    filters = config["sample"]["annual_filters"]
    chunks: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(csv_name) as handle:
            for chunk in pd.read_csv(handle, usecols=ANNUAL_COLUMNS, chunksize=100_000):
                for field in ("datadate", "pdate", "fdate"):
                    chunk[field] = pd.to_datetime(chunk[field], errors="coerce")
                mask = pd.Series(True, index=chunk.index)
                for column, value in filters.items():
                    mask &= chunk[column].eq(value)
                # Retain pre-sample years because multi-year signal operators need them,
                # but focus availability coverage on the requested reporting horizon.
                chunk = chunk.loc[mask & chunk["datadate"].between("1995-01-01", config["sample"]["end_date"])]
                chunks.append(chunk)
    annual = pd.concat(chunks, ignore_index=True)
    annual["has_pdate"] = annual["pdate"].notna()
    annual["has_fdate"] = annual["fdate"].notna()
    annual["uses_fdate"] = annual["has_fdate"]
    annual["uses_fallback"] = ~annual["has_fdate"]
    annual["availability_date"] = annual["fdate"].fillna(
        annual["datadate"] + pd.DateOffset(months=config["availability"]["missing_fdate_lag_months"])
    )
    annual["report_year"] = annual["datadate"].dt.year

    coverage = annual.groupby("report_year", as_index=False).agg(
        records=("gvkey", "size"),
        pdate_records=("has_pdate", "sum"),
        fdate_records=("has_fdate", "sum"),
        fallback_records=("uses_fallback", "sum"),
    )
    coverage["fdate_share"] = coverage["fdate_records"] / coverage["records"]

    # Exact duplicates can represent a revision; same-calendar-year records can
    # also arise from a fiscal-year change and must not automatically be called a revision.
    exact = annual.duplicated(["gvkey", "datadate"], keep=False)
    annual["duplicate_gvkey_datadate"] = exact
    calendar = annual.duplicated(["gvkey", "report_year"], keep=False)
    summary = {
        "records": int(len(annual)),
        "pdate_records": int(annual["has_pdate"].sum()),
        "fdate_records": int(annual["has_fdate"].sum()),
        "fallback_records": int(annual["uses_fallback"].sum()),
        "exact_duplicate_records": int(exact.sum()),
        "same_gvkey_calendar_year_records": int(calendar.sum()),
        "filters": filters,
        "popsrc_filter_applied": False,
    }
    coverage.to_csv(out / "annual_pit_coverage_by_report_year.csv", index=False)
    annual.loc[exact, ANNUAL_COLUMNS + ["availability_date"]].sort_values(
        ["gvkey", "datadate", "availability_date"]
    ).to_parquet(out / "annual_exact_duplicate_events.parquet", index=False)
    (out / "annual_pit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
