"""Backtest grid built from persisted narrow panels and DuckDB windows."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .duckdb_weights import connection, weight_query
from .industry_groups import sic_group_expression


LABELS = {
    "ConservatismAdjustedProfitLeanClaims": "Conservatism Adjusted Profit Lean Claims",
    "NegativeWorkingCapitalProfitSort": "Negative Working Capital Profit Sort",
    "PredictableAccrualsSustainedProfit": "Predictable Accruals Sustained Profit",
    "TroughCoverageCleanAccrualRollover": "Trough Coverage Clean Accrual Rollover",
    "WorkingCapitalBloatVsPeers": "Working Capital Bloat vs. Peers",
    "knowledge_capital_built_versus_bought": "Knowledge Capital Built Versus Bought",
    "purchased_share_of_growth_spending": "Purchased Share of Growth Spending",
}
METHODS = ("zscore", "rank", "quintile")
FREQUENCIES = ("daily", "weekly", "monthly")


def _portfolio_query(weights_sql: str, frequencies: tuple[str, ...] = FREQUENCIES) -> str:
    """Apply all three rebalancing calendars to one set of daily weights.

    Signal scores are calculated once.  The resulting query streams a wide
    nine-column holding state (3 frequencies x 3 weight rules), rather than
    looping through dates in Python or materializing a weight cache.
    """
    target_columns, held_columns, execution_columns, prior_columns = [], [], [], []
    for frequency in frequencies:
        is_formation = {
            "daily": "TRUE",
            "weekly": "c.date = c.week_end",
            "monthly": "c.date = c.month_end",
        }[frequency]
        for method in METHODS:
            key = f"{frequency}_{method}"
            target_columns.append(
                f"CASE WHEN {is_formation} THEN coalesce(w.{method}_weight, 0.0) END AS target_{key}"
            )
            held_columns.append(
                f"last_value(target_{key} IGNORE NULLS) OVER (PARTITION BY permno ORDER BY date) AS held_{key}"
            )
            execution_columns.append(f"lag(held_{key}, 2) OVER (PARTITION BY permno ORDER BY date) AS weight_{key}")
            prior_columns.append(f"lag(weight_{key}) OVER (PARTITION BY permno ORDER BY date) AS prior_{key}")

    aggregates = []
    for frequency in frequencies:
        for method in METHODS:
            key = f"{frequency}_{method}"
            aggregates.extend([
                f"sum(coalesce(weight_{key}, 0.0) * coalesce(ret, 0.0)) AS return_{key}",
                f"count(*) FILTER (WHERE abs(coalesce(weight_{key}, 0.0)) > 1e-12) AS bets_{key}",
                f"sum(abs(coalesce(weight_{key}, 0.0) - coalesce(prior_{key}, 0.0))) AS turnover_{key}",
            ])
    return f"""
    WITH weights AS ({weights_sql}),
    calendar AS (
      SELECT date,
        max(date) OVER (PARTITION BY date_trunc('week', date)) AS week_end,
        max(date) OVER (PARTITION BY date_trunc('month', date)) AS month_end
      FROM (SELECT DISTINCT date FROM stock)
    ), targets AS (
      -- Weights already contain every tradable, non-null-signal row.  Keeping
      -- the holding expansion on this relation avoids carrying the unrelated
      -- no-signal rows through nine window states.
      SELECT w.date, w.permno, w.ret, {', '.join(target_columns)}
      FROM weights AS w JOIN calendar AS c USING (date)
    ), held AS (
      SELECT *, {', '.join(held_columns)} FROM targets
    ), executed AS (
      SELECT *, {', '.join(execution_columns)} FROM held
    ), with_prior AS (
      SELECT *, {', '.join(prior_columns)} FROM executed
    )
    SELECT date, {', '.join(aggregates)}
    FROM with_prior GROUP BY date ORDER BY date
    """


def _long_results(wide: pd.DataFrame, cutoff: float, sic_level: int | None, signal: str, frequencies: tuple[str, ...]) -> pd.DataFrame:
    records = []
    for frequency in frequencies:
        for method in METHODS:
            key = f"{frequency}_{method}"
            records.append(pd.DataFrame({
                "date": wide["date"], "ls_return": wide[f"return_{key}"],
                "n_bets": wide[f"bets_{key}"], "two_way_turnover": wide[f"turnover_{key}"],
                "microcap_percentile": cutoff, "sic_level": sic_level,
                "signal": LABELS.get(signal, signal), "rebalance_frequency": frequency,
                "weight_method": method,
            }))
    return pd.concat(records, ignore_index=True)


def _metrics(frame: pd.DataFrame, nw_lags: int) -> dict:
    ret = frame["ls_return"].dropna()
    if ret.empty:
        return {"n_days": 0, "AnnRet": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "AvgDlyBets": np.nan, "2WayTurnover": np.nan, "MDD": np.nan, "t_stat": np.nan}
    wealth = (1 + ret).cumprod()
    model = sm.OLS(ret.to_numpy(), np.ones((len(ret), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    return {
        "n_days": len(ret), "AnnRet": wealth.iloc[-1] ** (252 / len(ret)) - 1,
        "AnnVol": ret.std(ddof=1) * np.sqrt(252),
        "Sharpe": ret.mean() / ret.std(ddof=1) * np.sqrt(252),
        "AvgDlyBets": frame["n_bets"].mean(), "2WayTurnover": frame["two_way_turnover"].mean(),
        "MDD": (wealth / wealth.cummax() - 1).min(), "t_stat": float(model.tvalues[0]),
    }


def run_panel_grid(config: dict) -> None:
    """Write one fresh cache containing the configured Cartesian product."""
    root = Path(config["paths"]["derived_root"])
    raw_signals = json.loads((root / "signals" / "annual_signal_events_manifest.json").read_text())["signals"]
    grid = config.get("grid", {})
    cuts = grid.get("microcap_percentiles", config["universe"]["microcap_percentiles"])
    sic_levels = grid.get("sic_levels", [config["portfolio"].get("sic_level")])
    all_results = []
    total = len(raw_signals) * len(cuts) * len(sic_levels)
    done = 0
    for level in sic_levels:
        scoped = copy.deepcopy(config)
        scoped["portfolio"]["sic_level"] = level
        score_group = sic_group_expression(scoped)
        group_weighting = scoped.get("industry_grouping", {}).get("group_weighting", "equal_group_gross")
        for signal in raw_signals:
            for cutoff in cuts:
                done += 1
                for frequency in FREQUENCIES:
                    con = connection(scoped, signal)
                    try:
                        wide = con.execute(_portfolio_query(weight_query(cutoff, score_group, group_weighting), (frequency,))).fetchdf()
                    finally:
                        con.close()
                    all_results.append(_long_results(wide, cutoff, level, signal, (frequency,)))
                print(f"completed {done}/{total}: sic={level}, cutoff={cutoff:.0%}, signal={signal}", flush=True)

    daily = pd.concat(all_results, ignore_index=True).sort_values(["rebalance_frequency", "weight_method", "sic_level", "microcap_percentile", "signal", "date"])
    daily["cum_return"] = daily.groupby(["rebalance_frequency", "weight_method", "sic_level", "microcap_percentile", "signal"], dropna=False)["ls_return"].transform(lambda values: (1 + values).cumprod() - 1)
    output = root / "backtest"
    output.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output / "daily_ls_returns.parquet", index=False)
    keys = ["rebalance_frequency", "weight_method", "sic_level", "microcap_percentile", "signal"]
    metrics = pd.DataFrame([{**dict(zip(keys, values)), **_metrics(group, config["portfolio"]["newey_west_lags"])} for values, group in daily.groupby(keys, dropna=False)])
    metrics.to_csv(output / "performance_metrics.csv", index=False)
    print(f"wrote {len(metrics)} portfolios to {output}", flush=True)
