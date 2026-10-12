"""Vectorised portfolio formation for cached PIT signal panels."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .cache import load_score, store_score
from .execution import DenseReturnEngine
from .groups import sic_division_panel
from .market_data import panel as crsp_panel
from .performance import performance_metrics
from .pipeline import ensure_panels, panel
from .recipes import get_recipe, signal_names
from .scoring import prepare_score, score_prepared_many


FREQUENCIES = ("daily", "weekly", "monthly")
WEIGHT_METHODS = ("zscore", "rank", "quintile")
SIC_LEVELS = (None, 1)


def _load_result_cache(
    output: Path,
    *,
    universe: object,
    names: tuple[str, ...],
    frequencies: tuple[str, ...],
    methods: tuple[str, ...],
    sic_levels: tuple[int | None, ...],
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load a completed result grid only when its declared settings match."""
    daily_path = output / "daily_ls_returns.parquet"
    metrics_path = output / "performance_metrics.csv"
    manifest_path = output / "manifest.json"
    if not all(path.is_file() for path in (daily_path, metrics_path, manifest_path)):
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "signals": list(names),
        "frequencies": list(frequencies),
        "weight_methods": list(methods),
        "sic_levels": list(sic_levels),
        "return_definition": "DlyRet minus equal-weight mean DlyRet among non-null raw-signal securities each day",
        "signal_to_return_sessions": 2,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    cached_universe = manifest.get("universe", {})
    for field in ("start", "end"):
        requested = getattr(universe, field, None)
        if requested is not None and cached_universe.get(field) != str(requested):
            return None
    requested_universe = {
        field: value
        for field in ("rank_lte", "quantile_lte", "cumpct_lte", "lag")
        if (value := getattr(universe, field, None)) is not None
    }
    cached_request = manifest.get("requested_universe")
    if cached_request is not None and cached_request != requested_universe:
        return None
    try:
        return pd.read_parquet(daily_path), pd.read_csv(metrics_path)
    except (OSError, ValueError):
        return None


def _formation_mask(index: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    if frequency == "daily":
        return np.ones(len(index), dtype=bool)
    period = "W-FRI" if frequency == "weekly" else "M"
    series = pd.Series(index, index=index)
    last = series.groupby(index.to_period(period)).transform("max")
    return np.asarray(index == pd.DatetimeIndex(last.to_numpy()))


def rebalance_targets(weights: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Hold formation weights and explicitly close absent names at formation.

    A NaN in a score panel means that signal information is unavailable; it is
    not a zero score.  At a rebalance, however, an old position whose name is
    absent from the new target must be closed.  We therefore use zeros only in
    the held-position state at formation dates, while preserving NaNs in the
    unexecuted daily score panel.
    """
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of {FREQUENCIES}")
    target = weights.copy()
    target.index = pd.DatetimeIndex(pd.to_datetime(target.index))
    if frequency == "daily":
        return target
    formation = _formation_mask(target.index, frequency)
    held = pd.DataFrame(np.nan, index=target.index, columns=target.columns)
    held.loc[formation, :] = target.loc[formation, :].fillna(0.0)
    return held.ffill()


def excess_returns(returns: pd.DataFrame, raw_signal: pd.DataFrame) -> pd.DataFrame:
    """Daily adjusted return less the equal-weight mean for observable signals."""
    observed = returns.where(raw_signal.notna())
    benchmark = observed.mean(axis=1)
    return returns.sub(benchmark, axis=0)


def _records(
    *,
    signal: str,
    frequency: str,
    method: str,
    sic_level: int | None,
    daily_return: pd.Series,
    weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric = performance_metrics(daily_return, weights)
    metric.insert(0, "signal", signal)
    metric.insert(1, "rebalance_frequency", frequency)
    metric.insert(2, "weight_method", method)
    metric.insert(3, "sic_level", sic_level)
    daily = pd.DataFrame({"date": daily_return.index, "ls_return": daily_return.to_numpy()})
    daily["cum_return"] = (1.0 + daily["ls_return"].fillna(0.0)).cumprod() - 1.0
    daily["signal"] = signal
    daily["rebalance_frequency"] = frequency
    daily["weight_method"] = method
    daily["sic_level"] = sic_level
    return daily, metric


def run(
    *,
    data: object,
    universe: object,
    output: str | Path,
    names: Iterable[str] | None = None,
    frequencies: Iterable[str] = FREQUENCIES,
    methods: Iterable[str] = WEIGHT_METHODS,
    sic_levels: Iterable[int | None] = SIC_LEVELS,
    refresh_raw: bool = False,
    refresh_results: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the selected formula × portfolio-setting result grid.

    Raw formula panels and CRSP panels are cached separately; grid expansion
    only performs in-memory daily matrix operations, never scans raw vendors.
    A complete matching result grid is cached too, so a repeat call only reads
    the daily-return and metric files. Set ``refresh_results`` to rebuild that
    grid, or ``refresh_raw`` to rebuild both source panels and results.
    """
    selected_names = tuple(signal_names() if names is None else names)
    selected_frequencies, selected_methods, selected_sic = tuple(frequencies), tuple(methods), tuple(sic_levels)
    unknown_frequency = set(selected_frequencies).difference(FREQUENCIES)
    unknown_method = set(selected_methods).difference(WEIGHT_METHODS)
    unknown_sic = set(selected_sic).difference(SIC_LEVELS)
    if unknown_frequency or unknown_method or unknown_sic:
        raise ValueError(f"Unsupported grid selection: frequency={unknown_frequency}, method={unknown_method}, sic={unknown_sic}")
    output = Path(output)
    if not refresh_raw and not refresh_results:
        cached_results = _load_result_cache(
            output,
            universe=universe,
            names=selected_names,
            frequencies=selected_frequencies,
            methods=selected_methods,
            sic_levels=selected_sic,
        )
        if cached_results is not None:
            return cached_results

    # Materialise all missing raw panels with one Annual-event query, then load
    # and release them one signal at a time.  This keeps the full eleven-signal
    # grid bounded by the largest single wide panel rather than their sum.
    resolved = ensure_panels(selected_names, data=data, universe=universe, refresh=refresh_raw)
    returns = crsp_panel("ret", data=data, universe=resolved)
    sic = crsp_panel("siccd", data=data, universe=resolved) if 1 in selected_sic else None
    groups_by_sic = {None: None, 1: sic_division_panel(sic) if sic is not None else None}

    daily_parts: list[pd.DataFrame] = []
    metric_parts: list[pd.DataFrame] = []
    for name in selected_names:
        signal = panel(name, data=data, universe=resolved)
        recipe = get_recipe(name)
        adjusted = excess_returns(returns, signal.values)
        return_engine = DenseReturnEngine(adjusted, report_start=resolved.report_start)
        prepared = prepare_score(signal, winsorize=0.01)
        for level in selected_sic:
            groups = groups_by_sic[level]
            targets = {
                method: None if refresh_raw else load_score(name, resolved, recipe.version, method, level)
                for method in selected_methods
            }
            missing = tuple(method for method, target in targets.items() if target is None)
            if missing:
                computed = score_prepared_many(prepared, groups=groups, methods=missing)
                for method, scored in computed.items():
                    targets[method] = scored.values
                    store_score(name, resolved, recipe.version, method, level, scored.values)
            for method, target in targets.items():
                assert target is not None
                for frequency in selected_frequencies:
                    held_target = rebalance_targets(target, frequency)
                    pnl = return_engine.realized(held_target)
                    daily, metrics = _records(
                        signal=name,
                        frequency=frequency,
                        method=method,
                        sic_level=level,
                        daily_return=pnl,
                        weights=held_target,
                    )
                    daily_parts.append(daily)
                    metric_parts.append(metrics)

    daily = pd.concat(daily_parts, ignore_index=True)
    metrics = pd.concat(metric_parts, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output / "daily_ls_returns.parquet", index=False)
    metrics.to_csv(output / "performance_metrics.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "signals": list(selected_names),
        "frequencies": list(selected_frequencies),
        "weight_methods": list(selected_methods),
        "sic_levels": list(selected_sic),
        "universe": {
            "start": str(resolved.start), "end": str(resolved.end),
            "state_start": str(resolved.state_start), "coverage_permnos": len(resolved.coverage_permnos),
        },
        "requested_universe": {
            field: value
            for field in ("rank_lte", "quantile_lte", "cumpct_lte", "lag")
            if (value := getattr(universe, field, None)) is not None
        },
        "return_definition": "DlyRet minus equal-weight mean DlyRet among non-null raw-signal securities each day",
        "signal_to_return_sessions": 2,
    }, indent=2, sort_keys=True))
    return daily, metrics
