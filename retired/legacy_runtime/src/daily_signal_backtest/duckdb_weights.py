"""Panel-driven, one-signal-at-a-time daily weight formation."""

from __future__ import annotations

from pathlib import Path

import duckdb

from .panel_store import _slug


def _path_literal(path: Path) -> str:
    return str(path).replace("'", "''")


def connection(config: dict, signal: str) -> duckdb.DuckDBPyConnection:
    """Join common persisted panels to exactly one signal panel.

    The views are lazy: DuckDB reads only projected Parquet columns, and this
    function cannot accidentally read the wide signal-construction panel.
    """
    root = Path(config["paths"]["derived_root"]) / "panels" / "values"
    paths = [
        root / "return.parquet", root / "lagged_market_cap.parquet",
        root / "universe_mask.parquet", root / "sic.parquet",
        root / f"signal_{_slug(signal)}.parquet",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing persisted panel(s): " + ", ".join(missing))
    ret, cap, mask, sic, signal_path = (_path_literal(path) for path in paths)
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA preserve_insertion_order=false")
    temporary_directory = root.parent / "duckdb_tmp"
    temporary_directory.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit = '{config.get('engine', {}).get('memory_limit', '4GB')}'")
    con.execute(f"SET temp_directory = '{_path_literal(temporary_directory)}'")
    con.execute(f"""
        CREATE VIEW stock AS
        SELECT r.date, r.permno, r.value AS ret, c.value AS lagged_market_cap,
               s.value AS signal_value, m.is_nyse, m.is_nonfinancial,
               m.has_return, i.value::INTEGER AS siccd
        FROM read_parquet('{ret}') AS r
        JOIN read_parquet('{cap}') AS c USING (date, permno)
        JOIN read_parquet('{signal_path}') AS s USING (date, permno)
        JOIN read_parquet('{mask}') AS m USING (date, permno)
        JOIN read_parquet('{sic}') AS i USING (date, permno)
    """)
    return con


def weight_query(cutoff: float, score_group: str = "0", group_weighting: str = "equal_group_gross") -> str:
    """Yield zscore/rank/quintile daily weights from the ``stock`` view.

    ``score_group`` is a configurable industry-key expression.  The group
    scoring definition and the allocation across groups are intentionally
    separate: equal_group_gross equalizes active groups, while pooled_gross
    gives groups gross exposure in proportion to their observations/scores.
    """
    if not 0 < cutoff < 1:
        raise ValueError("cutoff must be strictly between zero and one")
    if group_weighting not in {"equal_group_gross", "pooled_gross"}:
        raise ValueError("group_weighting must be 'equal_group_gross' or 'pooled_gross'")
    if group_weighting == "equal_group_gross":
        z_weight = "z_group_weight / nullif(sum(abs(z_group_weight)) OVER (PARTITION BY date), 0)"
        rank_weight = "rank_group_weight / nullif(sum(abs(rank_group_weight)) OVER (PARTITION BY date), 0)"
        quintile_weight = "quintile_group_weight / nullif(sum(abs(quintile_group_weight)) OVER (PARTITION BY date), 0)"
    else:
        z_weight = "z / nullif(sum(abs(z)) OVER (PARTITION BY date), 0)"
        rank_weight = "(pct_rank - .5) / nullif(sum(abs(pct_rank - .5)) OVER (PARTITION BY date), 0)"
        quintile_weight = "quintile_group_weight / nullif(sum(abs(quintile_group_weight)) OVER (PARTITION BY date), 0)"
    return f"""
    WITH cap AS (
      SELECT *, quantile_cont(lagged_market_cap, {cutoff})
        FILTER (WHERE is_nyse AND lagged_market_cap > 0)
        OVER (PARTITION BY date) AS cutoff_cap
      FROM stock
    ), eligible AS (
      SELECT date, permno, ret, signal_value::DOUBLE AS value, ({score_group}) AS score_group
      FROM cap
      WHERE has_return AND is_nonfinancial AND lagged_market_cap > cutoff_cap
        AND signal_value IS NOT NULL AND ({score_group}) IS NOT NULL
    ), bounds AS (
      SELECT *, quantile_cont(value, .01) OVER (PARTITION BY date, score_group) AS lo,
        quantile_cont(value, .99) OVER (PARTITION BY date, score_group) AS hi
      FROM eligible
    ), clipped AS (
      SELECT date, permno, ret, score_group, least(greatest(value, lo), hi) AS value FROM bounds
    ), scored AS (
      SELECT *,
        (value - avg(value) OVER (PARTITION BY date, score_group))
          / nullif(stddev_pop(value) OVER (PARTITION BY date, score_group), 0) AS z,
        rank() OVER (PARTITION BY date, score_group ORDER BY value)
          / count(*) OVER (PARTITION BY date, score_group) AS pct_rank
      FROM clipped
    ), normalized AS (
      SELECT *,
        z / nullif(sum(abs(z)) OVER (PARTITION BY date, score_group), 0) AS z_group_weight,
        (pct_rank - .5) / nullif(sum(abs(pct_rank - .5)) OVER (PARTITION BY date, score_group), 0) AS rank_group_weight,
        count(*) FILTER (WHERE pct_rank >= .8) OVER (PARTITION BY date, score_group) AS n_long,
        count(*) FILTER (WHERE pct_rank <= .2) OVER (PARTITION BY date, score_group) AS n_short
      FROM scored
    ), grouped AS (
      SELECT *, CASE WHEN pct_rank >= .8 THEN .5 / n_long WHEN pct_rank <= .2 THEN -.5 / n_short ELSE 0 END AS quintile_group_weight
      FROM normalized
    )
    SELECT date, permno, ret,
      {z_weight} AS zscore_weight,
      {rank_weight} AS rank_weight,
      {quintile_weight} AS quintile_weight
    FROM grouped
    """
