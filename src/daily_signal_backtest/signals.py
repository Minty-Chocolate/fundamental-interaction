from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd


REQUIRED_VARIABLES = [
    "at", "ceq", "ib", "sale", "revt", "cogs", "xsga", "xrd", "capx", "aqc", "dp", "ppegt", "ppent", "invt", "act", "lct", "dltt", "dlc", "txdi", "itcb", "pstkrv", "pstkl", "pstk", "txditc", "seq", "lt", "ni", "oibdp", "oiadp", "che", "rect", "aco", "ap", "xacc", "intan", "ao", "lo", "spi", "gdwl", "emp", "txp", "xint", "csho", "ajex", "prcc_f", "oancf", "dltis", "dltr", "sstk", "prstkc", "dvc", "dvt", "dpc", "recch", "invch", "apalch", "txach", "aoloch", "fopt", "pi", "txt", "ebitda", "xad", "re", "wcap", "mib", "sich",
]


def _read_annual(config: dict) -> pd.DataFrame:
    columns = sorted(set(REQUIRED_VARIABLES + ["gvkey", "datadate", "fyear", "fyr", "indfmt", "datafmt", "consol", "fdate"]))
    path = config["paths"]["compustat_annual_zip"]
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(csv_name) as handle:
            for chunk in pd.read_csv(handle, usecols=columns, chunksize=100_000, low_memory=False):
                for field in ("datadate", "fdate"):
                    chunk[field] = pd.to_datetime(chunk[field], errors="coerce")
                filters = config["sample"]["annual_filters"]
                keep = chunk["datadate"].between("1995-01-01", config["sample"]["end_date"])
                for column, value in filters.items():
                    keep &= chunk[column].eq(value)
                frames.append(chunk.loc[keep])
    annual = pd.concat(frames, ignore_index=True)
    annual["availability_date"] = annual["fdate"].fillna(
        annual["datadate"] + pd.DateOffset(months=config["availability"]["missing_fdate_lag_months"])
    )
    annual["availability_source"] = annual["fdate"].notna().map({True: "fdate", False: "fallback_6m"})
    annual["year"] = annual["datadate"].dt.year
    return annual


def _read_ccm(config: dict) -> pd.DataFrame:
    ccm = pd.read_csv(
        config["paths"]["ccm_csv"], dtype={"GVKEY": str, "LPERMNO": "Int64"}, low_memory=False
    )
    ccm.columns = [column.lower() for column in ccm.columns]
    ccm = ccm.loc[ccm["linktype"].isin(["LC", "LU"]) & ccm["linkprim"].isin(["P", "C"])]
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"], format="%d%b%Y", errors="coerce")
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"], format="%d%b%Y", errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    ccm = ccm.dropna(subset=["lpermno", "linkdt"])
    return ccm[["gvkey", "lpermno", "linkdt", "linkenddt"]].rename(columns={"lpermno": "permno"})


def build_annual_signal_events(config: dict) -> None:
    """Evaluate published expressions at annual frequency and attach availability dates."""
    repo_src = "/data/raw/signal-laboratory-main/src"
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)
    from siglab.agent.executor import SignalEngine

    annual = _read_annual(config)
    ccm = _read_ccm(config)
    panel = annual.merge(ccm, on="gvkey", how="inner")
    panel = panel.loc[panel["availability_date"].between(panel["linkdt"], panel["linkenddt"])]
    panel["permno"] = panel["permno"].astype(int)
    panel = panel.sort_values(["permno", "year", "datadate"]).drop_duplicates(["permno", "year"], keep="last")
    panel["sic"] = panel["sich"]
    # Retain the published expression engine's annual cross-sectional key.
    # Availability is deliberately handled by a separate event timestamp.
    panel["form_year"] = panel["year"] + 1
    dummy_crsp = pd.DataFrame(columns=["permno", "date", "month", "year"])
    engine = SignalEngine(comp=panel, crsp=dummy_crsp)
    catalog = json.loads(Path(config["paths"]["signal_catalog"]).read_text())
    events = engine.comp[["permno", "gvkey", "datadate", "fdate", "availability_date", "availability_source", "year"]].copy()
    for signal in catalog:
        events[signal["name"]] = engine._eval(signal["expression"]).astype(float).to_numpy()
    out = Path(config["paths"]["derived_root"]) / "signals"
    out.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out / "annual_signal_events.parquet", index=False)
    manifest = {
        "rows": int(len(events)),
        "signals": [signal["name"] for signal in catalog],
        "availability_rule": "fdate else datadate_plus_6_months",
        "pdate_used": False,
    }
    (out / "annual_signal_events_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
