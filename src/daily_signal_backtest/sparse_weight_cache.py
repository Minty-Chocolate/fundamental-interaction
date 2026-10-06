"""Persistent active-row weight caches.

The research panels are canonically dense ``date x permno`` arrays.  Score
formation only needs rows which can actually enter a portfolio, so this module
computes and stores that much smaller representation. It deliberately
partitions by calendar year: every operation is a daily cross-section, hence
the partition changes neither ranks nor winsorisation while bounding memory.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd

from .panel_store import _slug


def _literal(path: Path) -> str:
    return str(path).replace("'", "''")


def _cache_directory(config: dict, signal: str, cutoff: float, sic_level: int | None) -> Path:
    root = Path(config["engine"]["weight_cache_root"]) / config["engine"]["weight_cache_version"]
    return root / _slug(signal) / f"cutoff={cutoff:.2f}" / f"sic_level={'none' if sic_level is None else sic_level}"


def weight_cache_path(config: dict, signal: str, cutoff: float, sic_level: int | None) -> Path:
    """Return the cache directory rather than one monolithic parquet file."""
    return _cache_directory(config, signal, cutoff, sic_level)


def _universe_directory(config: dict, cutoff: float) -> Path:
    root = Path(config["engine"]["weight_cache_root"]) / config["engine"]["weight_cache_version"]
    return root / "_universe" / f"cutoff={cutoff:.2f}"


def _division_expression() -> str:
    # User-approved SIC divisions. Financials are removed before this mapping.
    return """CASE
      WHEN siccd BETWEEN 100 AND 999 THEN 0 WHEN siccd BETWEEN 1000 AND 1499 THEN 1
      WHEN siccd BETWEEN 1500 AND 1799 THEN 2 WHEN siccd BETWEEN 2000 AND 3999 THEN 3
      WHEN siccd BETWEEN 4000 AND 4999 THEN 4 WHEN siccd BETWEEN 5000 AND 5199 THEN 5
      WHEN siccd BETWEEN 5200 AND 5999 THEN 6 WHEN siccd BETWEEN 6000 AND 6799 THEN 7
      WHEN siccd BETWEEN 7000 AND 8999 THEN 8 WHEN siccd BETWEEN 9100 AND 9729 THEN 9
      WHEN siccd BETWEEN 9900 AND 9999 THEN 10 ELSE NULL END"""


def _universe_query(cutoff: float, start: str, stop: str, source_root: Path) -> str:
    """The signal-independent daily trade universe for one cutoff/year."""
    source = lambda name: _literal(source_root / f"{name}.parquet")
    return f"""
    WITH stock AS (
      SELECT r.date, r.permno, r.value AS ret, c.value AS cap, i.value::INTEGER AS siccd
      FROM read_parquet('{source('return')}') r
      JOIN read_parquet('{source('lagged_market_cap')}') c USING(date, permno)
      JOIN read_parquet('{source('sic')}') i USING(date, permno)
      WHERE r.date >= DATE '{start}' AND r.date < DATE '{stop}'
    ), cutoff_panel AS (
      SELECT *, quantile_cont(cap, {cutoff}) FILTER (
        WHERE ret IS NOT NULL AND cap > 0 AND NOT (siccd BETWEEN 6000 AND 6999)
      ) OVER (PARTITION BY date) AS cap_cutoff
      FROM stock
    )
    SELECT date, permno, siccd
    FROM cutoff_panel
    WHERE ret IS NOT NULL AND cap >= cap_cutoff AND NOT (siccd BETWEEN 6000 AND 6999)
    """


def _query(sic_level: int | None, universe_path: Path, source_root: Path, signal: str) -> str:
    """Score one signal against a precomputed eligible universe partition."""
    group = "0" if sic_level is None else _division_expression()
    source = lambda name: _literal(source_root / f"{name}.parquet")
    signal_path = source(f"signal_{_slug(signal)}")
    return f"""
    WITH eligible AS (
      SELECT u.date, u.permno, s.value::DOUBLE AS value, ({group}) AS score_group
      FROM read_parquet('{_literal(universe_path)}') u
      JOIN read_parquet('{signal_path}') s USING(date, permno)
      WHERE s.value IS NOT NULL AND ({group}) IS NOT NULL
    ), clipped AS (
      SELECT *,
        least(
          greatest(value, quantile_cont(value, .01) OVER (PARTITION BY date)),
          quantile_cont(value, .99) OVER (PARTITION BY date)
        ) AS x
      FROM eligible
    ), ranked AS (
      SELECT *,
        avg(x) OVER (PARTITION BY date, score_group) AS group_mean,
        stddev_pop(x) OVER (PARTITION BY date, score_group) AS group_std,
        rank() OVER (PARTITION BY date, score_group ORDER BY x) AS group_rank,
        count(*) OVER (PARTITION BY date, score_group) AS group_count
      FROM clipped
    ), raw AS (
      SELECT date, permno,
        (x - group_mean) / nullif(group_std, 0) AS z_raw,
        group_rank / group_count - .5 AS rank_raw,
        CASE WHEN group_rank / group_count >= .8 THEN 1.0
             WHEN group_rank / group_count <= .2 THEN -1.0 ELSE 0.0 END AS quintile_raw
      FROM ranked
    ), centered AS (
      SELECT date, permno,
        z_raw - avg(z_raw) OVER (PARTITION BY date) AS z,
        rank_raw - avg(rank_raw) OVER (PARTITION BY date) AS r,
        quintile_raw - avg(quintile_raw) OVER (PARTITION BY date) AS q
      FROM raw
    )
    SELECT date, permno,
      z / nullif(sum(abs(z)) OVER (PARTITION BY date), 0) AS zscore_weight,
      r / nullif(sum(abs(r)) OVER (PARTITION BY date), 0) AS rank_weight,
      q / nullif(sum(abs(q)) OVER (PARTITION BY date), 0) AS quintile_weight
    FROM centered
    """


def _local_input_root(config: dict) -> Path:
    return Path(config["engine"]["weight_cache_root"]) / "input_values"


def mirror_value_inputs(config: dict) -> Path:
    """Mirror the small columnar source once from T7 to the local SSD."""
    source = Path(config["paths"]["derived_root"]) / "panels" / "values"
    destination = _local_input_root(config)
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.glob("*.parquet"):
        target = destination / item.name
        if not target.exists() or target.stat().st_size != item.stat().st_size:
            shutil.copy2(item, target)
    return destination


def _years(config: dict) -> list[int]:
    sample = config["sample"]
    return list(range(pd.Timestamp(sample["start_date"]).year, pd.Timestamp(sample["end_date"]).year + 1))


def _manifest(config: dict, signal: str, cutoff: float, sic_level: int | None, years: list[int]) -> dict:
    return {
        "version": config["engine"]["weight_cache_version"],
        "signal": signal,
        "cutoff": cutoff,
        "sic_level": sic_level,
        "years": years,
        "semantics": "global winsorisation; SIC-internal raw scoring; global net=0/gross=1",
    }


def _is_complete(output: Path, manifest: dict) -> bool:
    path = output / "manifest.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()) == manifest and all((output / f"year={year}.parquet").exists() for year in manifest["years"])
    except json.JSONDecodeError:
        return False


def _materialize_universe_year(config: dict, cutoff: float, year: int, source_root: Path) -> Path:
    output = _universe_directory(config, cutoff) / f"year={year}.parquet"
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.parquet")
    partial.unlink(missing_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={config['engine'].get('cache_threads_per_worker', 2)}")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute(
        f"COPY ({_universe_query(cutoff, f'{year}-01-01', f'{year + 1}-01-01', source_root)}) "
        f"TO '{_literal(partial)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    partial.replace(output)
    print(f"cached universe, cutoff={cutoff:.0%}, year={year}", flush=True)
    return output


def materialize_universe_caches(config: dict, cutoffs: list[float]) -> None:
    """Build each cutoff's signal-independent universe once, on the local SSD."""
    source_root = mirror_value_inputs(config)
    jobs = [(cutoff, year) for cutoff in cutoffs for year in _years(config)]
    workers = config["engine"].get("cache_workers", 3)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_materialize_universe_year, config, cutoff, year, source_root): (cutoff, year) for cutoff, year in jobs}
        for complete, future in enumerate(as_completed(futures), start=1):
            future.result()
            if complete % 10 == 0 or complete == len(jobs):
                print(f"universe cache {complete}/{len(jobs)}", flush=True)
def materialize_sparse_weights(config: dict, signal: str, cutoff: float, sic_level: int | None) -> Path:
    """Calculate score weights once, in independent one-year cache partitions."""
    output = _cache_directory(config, signal, cutoff, sic_level)
    years = _years(config)
    manifest = _manifest(config, signal, cutoff, sic_level, years)
    if _is_complete(output, manifest):
        return output
    source_root = mirror_value_inputs(config)
    output.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={config['engine'].get('cache_threads_per_worker', 2)}")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute("PRAGMA memory_limit='6GB'")
    for year in years:
        destination = output / f"year={year}.parquet"
        if destination.exists():
            continue
        partial = output / f"year={year}.partial.parquet"
        partial.unlink(missing_ok=True)
        universe_path = _materialize_universe_year(config, cutoff, year, source_root)
        con.execute(
            f"COPY ({_query(sic_level, universe_path, source_root, signal)}) "
            f"TO '{_literal(partial)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        partial.replace(destination)
        print(f"cached {signal}, cutoff={cutoff:.0%}, sic_level={sic_level}, year={year}", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return output
