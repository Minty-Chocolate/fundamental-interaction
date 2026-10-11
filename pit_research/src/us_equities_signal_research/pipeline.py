"""Daily PIT raw-panel dispatcher; deliberately independent of legacy code."""

from __future__ import annotations

from datetime import date

import pandas as pd

from .cache import load_raw, store_raw
from .model import FundamentalEvents, RawSignalPanel
from .recipes import get_recipe


def _years_before(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:  # 29 February
        return value.replace(year=value.year - years, day=28)


def panel(
    name: str,
    *,
    data: object,
    universe: object,
    refresh: bool = False,
) -> RawSignalPanel:
    """Materialise one recipe's raw daily panel on the CRSP trading calendar.

    ``universe`` is a dynamic rule.  It is resolved once into daily membership
    plus two opening state sessions; its coverage union constrains data reads
    but never substitutes for the daily mask used later by ``score``.
    """
    recipe = get_recipe(name)
    resolved = data.universe.resolve(universe, opening_state_sessions=2)
    cached = None if refresh else load_raw(recipe.name, resolved, recipe.version)
    if cached is not None:
        values, path = cached
        return RawSignalPanel(name=recipe.name, values=values, universe=resolved, cache_path=path)

    prefetch_years = recipe.annual_lookback_years + recipe.prefetch_buffer_years
    prefetch_start = _years_before(resolved.start, prefetch_years)
    annual = data.compustat.annual.events_for_universe(
        recipe.required_fields,
        universe=resolved,
        ccm=data.ccm,
        prefetch_start=prefetch_start,
    )
    calendar = pd.DatetimeIndex(pd.to_datetime(resolved.calendar.get_column("date").to_list()), name="date")
    events = FundamentalEvents(annual=annual, data=data, coverage_permnos=resolved.coverage_permnos)
    values = recipe.compute(events, calendar)
    values = values.reindex(index=calendar, columns=pd.Index(resolved.coverage_permnos, dtype="int64"))
    path = store_raw(
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
        },
    )
    return RawSignalPanel(name=recipe.name, values=values, universe=resolved, cache_path=path)
