"""Daily PIT raw-panel materialisation and cache dispatch."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import pandas as pd

from .cache import load_raw, store_raw
from .model import FundamentalEvents, RawSignalPanel
from .recipes import compute_recipes, get_recipe


def _years_before(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:  # 29 February
        return value.replace(year=value.year - years, day=28)


def _resolved(data: object, universe: object) -> object:
    return universe if hasattr(universe, "membership") else data.universe.resolve(universe, opening_state_sessions=2)


def ensure_panels(
    names: Iterable[str],
    *,
    data: object,
    universe: object,
    refresh: bool = False,
) -> object:
    """Materialise any missing raw panels, returning the resolved universe.

    Missing recipes share exactly one Annual-event read.  Cache hits are not
    deserialised here: callers that process signals one at a time can keep only
    one wide raw panel resident in memory.
    """
    recipes = [get_recipe(name) for name in names]
    if not recipes:
        return _resolved(data, universe)
    resolved = _resolved(data, universe)
    missing = [
        recipe
        for recipe in recipes
        if refresh or load_raw(recipe.name, resolved, recipe.version) is None
    ]
    if not missing:
        return resolved

    prefetch_years = max(recipe.annual_lookback_years + recipe.prefetch_buffer_years for recipe in missing)
    prefetch_start = _years_before(resolved.start, prefetch_years)
    fields = tuple(sorted({field for recipe in missing for field in recipe.required_fields}))
    annual = data.compustat.annual.events_for_universe(
        fields,
        universe=resolved,
        ccm=data.ccm,
        prefetch_start=prefetch_start,
    )
    calendar = pd.DatetimeIndex(pd.to_datetime(resolved.calendar.get_column("date").to_list()), name="date")
    events = FundamentalEvents(
        annual=annual, data=data, coverage_permnos=resolved.coverage_permnos,
        annual_pandas=annual.to_pandas(),
    )
    values_by_recipe = compute_recipes(tuple(missing), events, calendar)
    for recipe in missing:
        values = values_by_recipe[recipe.name]
        store_raw(
            recipe.name,
            resolved,
            recipe.version,
            values,
            {
                "required_fields": list(recipe.required_fields),
                "annual_lookback_years": recipe.annual_lookback_years,
                "prefetch_buffer_years": recipe.prefetch_buffer_years,
                "prefetch_start": str(prefetch_start),
                "source_events": annual.height,
                "batched_formula_count": len(missing),
            },
        )
    return resolved


def panels(
    names: Iterable[str],
    *,
    data: object,
    universe: object,
    refresh: bool = False,
) -> dict[str, RawSignalPanel]:
    """Load several materialised raw panels into memory.

    Use :func:`ensure_panels` plus :func:`panel` when memory bounded
    signal-by-signal processing is preferable.
    """
    recipes = [get_recipe(name) for name in names]
    resolved = ensure_panels((recipe.name for recipe in recipes), data=data, universe=universe, refresh=refresh)
    output: dict[str, RawSignalPanel] = {}
    for recipe in recipes:
        cached = load_raw(recipe.name, resolved, recipe.version)
        if cached is None:  # defensive: ensure_panels must have produced it
            raise RuntimeError(f"Raw panel materialisation did not produce {recipe.name!r}.")
        values, path = cached
        output[recipe.name] = RawSignalPanel(name=recipe.name, values=values, universe=resolved, cache_path=path)
    return output


def panel(
    name: str,
    *,
    data: object,
    universe: object,
    refresh: bool = False,
) -> RawSignalPanel:
    """Load one raw daily ``date × permno`` signal panel from its cache."""
    recipe = get_recipe(name)
    resolved = ensure_panels((recipe.name,), data=data, universe=universe, refresh=refresh)
    cached = load_raw(recipe.name, resolved, recipe.version)
    if cached is None:  # defensive: ensure_panels must have produced it
        raise RuntimeError(f"Raw panel materialisation did not produce {recipe.name!r}.")
    values, path = cached
    return RawSignalPanel(name=recipe.name, values=values, universe=resolved, cache_path=path)
