"""Daily, weekly, and monthly LS portfolios using cached active-row weights.

Canonical inputs remain compact-wide date-by-permno panels on the local SSD.
Weight formation is sparse and persistent because it only contains stocks that
can enter a given portfolio.  The runner scatters one score at a time into RAM
and never creates a dense weight file or a dense held-position file on disk.
"""

from __future__ import annotations

import gc
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .sparse_weight_cache import materialize_sparse_weights, materialize_universe_caches


METHODS = ("zscore", "rank", "quintile")
FREQUENCIES = ("daily", "weekly", "monthly")
LABELS = {
    "Broad Universe Profitable Value Model": "Broad Universe Profitable Value Model",
    "ConservatismAdjustedProfitLeanClaims": "Conservatism Adjusted Profit Lean Claims",
    "Funded Research Tilt With Profit Delivery": "Funded Research Tilt With Profit Delivery",
    "Gross Margin Inventory Efficiency": "Gross Margin Inventory Efficiency",
    "NegativeWorkingCapitalProfitSort": "Negative Working Capital Profit Sort",
    "PredictableAccrualsSustainedProfit": "Predictable Accruals Sustained Profit",
    "Sustained Coverage With Capped Cushion": "Sustained Coverage With Capped Cushion",
    "TroughCoverageCleanAccrualRollover": "Trough Coverage Clean Accrual Rollover",
    "WorkingCapitalBloatVsPeers": "Working Capital Bloat vs. Peers",
    "knowledge_capital_built_versus_bought": "Knowledge Capital Built Versus Bought",
    "purchased_share_of_growth_spending": "Purchased Share of Growth Spending",
}


def _formation_dates(dates: np.ndarray, frequency: str) -> np.ndarray:
    date = pd.DatetimeIndex(dates)
    if frequency == "daily":
        return np.ones(len(date), dtype=bool)
    period = date.to_period("W-FRI" if frequency == "weekly" else "M")
    return np.r_[period[:-1] != period[1:], True]


def _market_excess_return(ret: np.memmap, cap: np.memmap, sic: np.memmap) -> np.ndarray:
    """Lag-cap-weighted broad nonfinancial market return, calculated in blocks."""
    result = np.full(ret.shape[0], np.nan, dtype=np.float64)
    for left in range(0, len(result), 64):
        right = min(left + 64, len(result))
        r = np.asarray(ret[left:right], dtype=np.float64)
        c = np.asarray(cap[left:right], dtype=np.float64)
        i = np.asarray(sic[left:right], dtype=np.float64)
        valid = np.isfinite(r) & np.isfinite(c) & (c > 0) & ~((i >= 6000) & (i <= 6999))
        numerator = np.sum(np.where(valid, r * c, 0.0), axis=1)
        denominator = np.sum(np.where(valid, c, 0.0), axis=1)
        result[left:right] = np.divide(numerator, denominator, out=np.full(right - left, np.nan), where=denominator > 0)
    return result


def _scatter_weights(cache: Path, dates: np.ndarray, permnos: np.ndarray) -> np.ndarray:
    """Load all three sparse scores into one temporary in-RAM wide tensor."""
    score = np.full((len(dates), len(permnos), len(METHODS)), np.nan, dtype=np.float32)
    date_lookup = pd.Index(pd.to_datetime(dates))
    permno_lookup = pd.Index(permnos)
    columns = [f"{method}_weight" for method in METHODS]
    for path in sorted(cache.glob("year=*.parquet")):
        frame = pd.read_parquet(path, columns=["date", "permno", *columns])
        row = date_lookup.get_indexer(pd.to_datetime(frame["date"]))
        col = permno_lookup.get_indexer(frame["permno"].astype(np.int64))
        if (row < 0).any() or (col < 0).any():
            raise ValueError(f"weight cache axis mismatch in {path}")
        score[row, col] = frame[columns].to_numpy(dtype=np.float32, na_value=np.nan)
    return score


def _portfolio_series(score: np.ndarray, ret: np.memmap, excess_market: np.ndarray, dates: np.ndarray, frequency: str, tolerance: float) -> dict[str, pd.DataFrame]:
    """Apply signal-t, execute-t+1 timing without materialising held weights.

    At each formation date the full score row replaces the old one, including
    NaNs. Thus a missing signal does not silently retain an old holding.
    """
    n_dates = len(dates)
    formation = _formation_dates(dates, frequency)
    anchors = np.maximum.accumulate(np.where(formation, np.arange(n_dates), -1))
    pnl = np.full((n_dates, len(METHODS)), np.nan, dtype=np.float64)
    bets = np.full((n_dates, len(METHODS)), np.nan, dtype=np.float64)
    turnover = np.full((n_dates, len(METHODS)), np.nan, dtype=np.float64)
    for left in range(1, n_dates, 32):
        right = min(left + 32, n_dates)
        execution_rows = anchors[left - 1:right - 1]
        previous_rows = anchors[left - 2:right - 2] if left >= 2 else np.r_[-1, anchors[:right - 2]]
        valid_execution = execution_rows >= 0
        valid_previous = previous_rows >= 0
        execution = np.full((right - left, score.shape[1], len(METHODS)), np.nan, dtype=np.float32)
        prior = np.full_like(execution, np.nan)
        execution[valid_execution] = score[execution_rows[valid_execution]]
        prior[valid_previous] = score[previous_rows[valid_previous]]
        r = np.asarray(ret[left:right], dtype=np.float64) - excess_market[left:right, None]
        pnl[left:right] = np.nansum(execution * r[:, :, None], axis=1)
        has_weight = np.isfinite(execution).any(axis=1)
        pnl[left:right][~valid_execution] = np.nan
        pnl[left:right][~has_weight] = np.nan
        bets[left:right] = np.count_nonzero(np.abs(execution) > tolerance, axis=1)
        bets[left:right][~valid_execution] = np.nan
        turnover[left:right] = np.sum(np.abs(np.nan_to_num(execution) - np.nan_to_num(prior)), axis=1)
        turnover[left:right][~(valid_execution & valid_previous)] = np.nan
    return {
        method: pd.DataFrame({"date": dates, "ls_return": pnl[:, index], "n_bets": bets[:, index], "two_way_turnover": turnover[:, index]})
        for index, method in enumerate(METHODS)
    }


def _metrics(frame: pd.DataFrame, nw_lags: int) -> dict:
    ret = frame["ls_return"].dropna().astype(float)
    if len(ret) < 2:
        return {"n_days": len(ret), "AnnRet": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "AvgDlyBets": np.nan, "2WayTurnover": np.nan, "MDD": np.nan, "t_stat": np.nan}
    wealth = (1 + ret).cumprod()
    model = sm.OLS(ret.to_numpy(), np.ones((len(ret), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": min(nw_lags, len(ret) - 1)})
    return {
        "n_days": len(ret),
        "AnnRet": wealth.iloc[-1] ** (252 / len(ret)) - 1,
        "AnnVol": ret.std(ddof=1) * np.sqrt(252),
        "Sharpe": ret.mean() / ret.std(ddof=1) * np.sqrt(252),
        "AvgDlyBets": frame["n_bets"].mean(),
        "2WayTurnover": frame["two_way_turnover"].mean(),
        "MDD": (wealth / wealth.cummax() - 1).min(),
        "t_stat": float(model.tvalues[0]),
    }


def _write_daily_part(path: Path, series: pd.DataFrame, signal: str, cutoff: float, sic_level: int | None, frequency: str, method: str) -> None:
    series["signal"] = LABELS.get(signal, signal)
    series["microcap_percentile"] = cutoff
    series["sic_level"] = sic_level
    series["rebalance_frequency"] = frequency
    series["weight_method"] = method
    series.to_parquet(path, index=False)


def _finalise(staging: Path, output: Path, nw_lags: int) -> None:
    daily = pd.concat((pd.read_parquet(path) for path in sorted((staging / "daily_parts").glob("*.parquet"))), ignore_index=True)
    keys = ["signal", "microcap_percentile", "sic_level", "rebalance_frequency", "weight_method"]
    daily = daily.sort_values([*keys, "date"])
    daily["cum_return"] = daily.groupby(keys, dropna=False)["ls_return"].transform(lambda x: (1 + x.fillna(0.0)).cumprod() - 1)
    daily.to_parquet(staging / "daily_ls_returns.parquet", index=False)
    metrics = pd.DataFrame([
        {**dict(zip(keys, values)), **_metrics(frame, nw_lags)}
        for values, frame in daily.groupby(keys, dropna=False)
    ])
    metrics.to_csv(staging / "performance_metrics.csv", index=False)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("daily_ls_returns.parquet", "performance_metrics.csv"):
        (staging / name).replace(output / name)
    shutil.rmtree(staging)


def _build_weight_caches(config: dict, signals: list[str], cutoffs: list[float], levels: list[int | None]) -> None:
    """Build independent score caches concurrently, once per signal/cutoff/SIC."""
    materialize_universe_caches(config, cutoffs)
    jobs = [(signal, cutoff, sic_level) for signal in signals for cutoff in cutoffs for sic_level in levels]
    workers = config["engine"].get("cache_workers", 3)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(materialize_sparse_weights, config, *job): job for job in jobs}
        for complete, future in enumerate(as_completed(futures), start=1):
            signal, cutoff, sic_level = futures[future]
            future.result()
            print(f"weight cache {complete}/{len(jobs)}: signal={signal}, cutoff={cutoff:.0%}, sic_level={sic_level}", flush=True)


def run_wide_grid(config: dict) -> None:
    """Run the Cartesian signal/cutoff/SIC grid from reusable score caches."""
    root = Path(config["paths"]["derived_root"])
    wide_root = Path(config["engine"].get("wide_cache_root", root / "wide"))
    output = root / "backtest"
    staging = root / "backtest.staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "daily_parts").mkdir(parents=True)
    axes = np.load(wide_root / "axes.npz")
    dates, permnos = axes["dates"], axes["permnos"]
    ret = np.load(wide_root / "return.npy", mmap_mode="r")
    cap = np.load(wide_root / "lagged_market_cap.npy", mmap_mode="r")
    sic = np.load(wide_root / "sic.npy", mmap_mode="r")
    excess_market = _market_excess_return(ret, cap, sic)
    signals = config["grid"].get("signals") or json.loads((root / "signals" / "annual_signal_events_manifest.json").read_text())["signals"]
    cutoffs, levels = config["grid"]["microcap_percentiles"], config["grid"]["sic_levels"]
    _build_weight_caches(config, signals, cutoffs, levels)
    total, done, part = len(signals) * len(cutoffs) * len(levels), 0, 0
    for signal in signals:
        for cutoff in cutoffs:
            for sic_level in levels:
                cache = materialize_sparse_weights(config, signal, cutoff, sic_level)
                score = _scatter_weights(cache, dates, permnos)
                for frequency in FREQUENCIES:
                    series_by_method = _portfolio_series(score, ret, excess_market, dates, frequency, config["portfolio"]["nonzero_weight_tolerance"])
                    for method, series in series_by_method.items():
                        _write_daily_part(staging / "daily_parts" / f"part-{part:04d}.parquet", series, signal, cutoff, sic_level, frequency, method)
                        part += 1
                del score
                gc.collect()
                done += 1
                print(f"completed {done}/{total}: signal={signal}, cutoff={cutoff:.0%}, sic_level={sic_level}", flush=True)
    _finalise(staging, output, config["portfolio"]["newey_west_lags"])
