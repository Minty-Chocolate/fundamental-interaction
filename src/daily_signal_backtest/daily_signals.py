from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


def materialize_daily_signals(config: dict) -> None:
    """ASOF-join annual signal events to daily CRSP observations.

    The result is a wide ``date × permno`` table.  No financial, microcap, or
    signal-missing screen is applied here; those belong to portfolio formation.
    """
    root = Path(config["paths"]["derived_root"])
    base_glob = str(root / "daily_stock_base" / "part-*.parquet")
    events = str(root / "signals" / "annual_signal_events.parquet")
    out = root / "daily_signals"
    if out.exists():
        raise RuntimeError(f"{out} exists; remove it only to intentionally rematerialize daily signals")
    out.mkdir(parents=True)
    con = duckdb.connect()
    # This host has 16GB RAM.  Force external spill rather than allowing one
    # sort/join to consume most of it and drive the OS into swap.
    con.execute("SET memory_limit='2.5GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"CREATE VIEW base AS SELECT * FROM read_parquet('{base_glob}', union_by_name=true)")
    con.execute(f"CREATE VIEW events AS SELECT * FROM read_parquet('{events}')")
    audits = []
    for year in range(pd.Timestamp(config["sample"]["start_date"]).year, pd.Timestamp(config["sample"]["end_date"]).year + 1):
        next_year = year + 1
        con.execute(f"CREATE OR REPLACE TEMP VIEW yearly AS SELECT * FROM base WHERE date >= DATE '{year}-01-01' AND date < DATE '{next_year}-01-01'")
        audits.append(con.execute("""
            SELECT count(*) AS duplicate_rows, count(*) AS duplicate_keys,
                   sum(CASE WHEN min_ret <> max_ret THEN 1 ELSE 0 END) AS inconsistent_return_keys
            FROM (SELECT permno, date, min(ret) AS min_ret, max(ret) AS max_ret FROM yearly GROUP BY permno, date HAVING count(*) > 1)
        """).fetchdf())
        con.execute("""
            CREATE OR REPLACE TEMP VIEW dedup AS
            SELECT * EXCLUDE (row_number) FROM (
              SELECT *, row_number() OVER (PARTITION BY permno, date ORDER BY delisting_flag DESC NULLS LAST) AS row_number FROM yearly
            ) WHERE row_number = 1
        """)
        part = out / f"year={year}"
        part.mkdir()
        con.execute(f"""COPY (SELECT b.*, e.* EXCLUDE (permno, availability_date) FROM dedup b ASOF LEFT JOIN events e ON b.permno = e.permno AND b.date >= e.availability_date) TO '{part / 'data.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
        print(f"wrote year={year}", flush=True)
    pd.concat(audits, ignore_index=True).to_csv(root / "daily_stock_base" / "duplicate_audit.csv", index=False)
    counts = con.execute(f"SELECT count(*) AS rows, count(DISTINCT permno) AS permnos FROM read_parquet('{out}/**/*.parquet')").fetchdf().iloc[0].to_dict()
    manifest = {"rows": int(counts["rows"]), "permnos": int(counts["permnos"]), "format": "wide", "annual_asof_key": "availability_date"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
