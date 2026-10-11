"""Independent PIT implementations of published signal definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from .model import FundamentalEvents, Recipe


def _last_non_null(values: pd.Series) -> float:
    observed = values.dropna().unique()
    if len(observed) == 0:
        return np.nan
    if len(observed) == 1:
        return float(observed[0])
    # Simultaneous conflicting revisions have no observable intraday order.
    # Treating them as unknown avoids arbitrarily selecting a future version.
    return np.nan


def _knowledge_component_events(annual: pd.DataFrame) -> pd.DataFrame:
    """Recompute firm history after each observable Annual event.

    This is intentionally done while records are still GVKEY keyed.  A late
    revision to an older fiscal period can therefore update a trailing window
    when that revision becomes observable, without changing prior dates.
    """
    fields = ("at", "oancf", "xrd", "intan")
    work = annual[["gvkey", "available_date", "fiscal_period_end", *fields]].copy()
    work["gvkey"] = work["gvkey"].astype(str).str.zfill(6)
    work["available_date"] = pd.to_datetime(work["available_date"])
    work["fiscal_period_end"] = pd.to_datetime(work["fiscal_period_end"])
    work = work.dropna(subset=["gvkey", "available_date", "fiscal_period_end"])
    for field in fields:
        work[field] = pd.to_numeric(work[field], errors="coerce")
    work = work.sort_values(["gvkey", "available_date", "fiscal_period_end"])

    records: list[dict[str, object]] = []
    for gvkey, firm in work.groupby("gvkey", sort=False):
        history: dict[pd.Timestamp, dict[str, float]] = {}
        for available_date, arrivals in firm.groupby("available_date", sort=False):
            # One availability date can disclose/revise several fiscal years.
            for fiscal_end, revision in arrivals.groupby("fiscal_period_end", sort=False):
                row = history.setdefault(pd.Timestamp(fiscal_end), {})
                for field in fields:
                    value = _last_non_null(revision[field])
                    if np.isfinite(value):
                        row[field] = value
            fiscal_dates = sorted(history)
            latest = history[fiscal_dates[-1]]
            assets = latest.get("at", np.nan)
            if not np.isfinite(assets) or abs(assets) <= 1e-12:
                continue

            ratios: list[float] = []
            for fiscal_end in fiscal_dates[-3:]:
                statement = history[fiscal_end]
                denominator = statement.get("at", np.nan)
                cash_flow = statement.get("oancf", np.nan)
                if np.isfinite(denominator) and abs(denominator) > 1e-12 and np.isfinite(cash_flow):
                    ratios.append(cash_flow / denominator)
            profitability = float(np.mean(ratios)) if ratios else np.nan

            research: list[float] = []
            for fiscal_end in fiscal_dates[-5:]:
                statement = history[fiscal_end]
                if "xrd" in statement:
                    research.append(statement["xrd"])
                elif "at" in statement:
                    # The published expression coalesces missing R&D to at-at.
                    research.append(0.0)
            built = float(np.sum(research) / assets) if research else np.nan
            purchased = latest.get("intan", 0.0)
            purchased = 0.0 if not np.isfinite(purchased) else purchased
            records.append(
                {
                    "gvkey": gvkey,
                    "available_date": available_date,
                    "cash_profitability": profitability,
                    "built_minus_bought": built - purchased / assets if np.isfinite(built) else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def _asof_component_panel(
    mapped: pd.DataFrame,
    *,
    component: str,
    calendar: pd.DatetimeIndex,
    permnos: tuple[int, ...],
) -> pd.DataFrame:
    """Expand one CCM-mapped sparse event component to a daily P-wide panel."""
    base = pd.DataFrame(index=calendar, columns=pd.Index(permnos, dtype="int64"), dtype="float64")
    if mapped.empty:
        return base
    work = mapped[["available_date", "permno", component]].copy()
    work["available_date"] = pd.to_datetime(work["available_date"])
    work["permno"] = work["permno"].astype("int64")
    work = work.loc[work["permno"].isin(permnos)].dropna(subset=["available_date"])
    if work.empty:
        return base
    # More than one GVKEY mapping to a cell is deliberately left unknown when
    # values conflict; we never choose an arbitrary source row.
    collapsed = (
        work.groupby(["available_date", "permno"], sort=True)[component]
        .agg(_last_non_null)
        .reset_index()
    )
    event_panel = collapsed.pivot(index="available_date", columns="permno", values=component)
    event_panel = event_panel.reindex(columns=base.columns)
    # Retaining events before state_start supplies the opening state without
    # reporting a pre-sample return.
    return event_panel.reindex(event_panel.index.union(calendar)).sort_index().ffill().reindex(calendar)


@dataclass(frozen=True)
class KnowledgeCapitalBuiltVersusBought:
    """Published B1 knowledge-capital signal, rebuilt under PIT timing."""

    name: str = "knowledge_capital_built_versus_bought"
    required_fields: tuple[str, ...] = ("at", "oancf", "xrd", "intan")
    annual_lookback_years: int = 5
    prefetch_buffer_years: int = 2
    version: str = "pit_daily_v1"

    def compute(self, events: FundamentalEvents, calendar: pd.DatetimeIndex) -> pd.DataFrame:
        annual = events.annual.to_pandas()
        if annual.empty:
            return pd.DataFrame(index=calendar, columns=pd.Index(events.coverage_permnos, dtype="int64"), dtype="float64")
        components = _knowledge_component_events(annual)
        if components.empty:
            return pd.DataFrame(index=calendar, columns=pd.Index(events.coverage_permnos, dtype="int64"), dtype="float64")
        mapped = events.data.ccm.map_events(pl.from_pandas(components)).to_pandas()
        cash = _asof_component_panel(
            mapped,
            component="cash_profitability",
            calendar=calendar,
            permnos=events.coverage_permnos,
        )
        build = _asof_component_panel(
            mapped,
            component="built_minus_bought",
            calendar=calendar,
            permnos=events.coverage_permnos,
        )
        # The paper's expression is SUMRANK(MA(OANCF/AT, 3),
        # WINSOR(SUM_5Y(XRD)/AT - INTAN/AT, 1)).  The inner 1% winsorization
        # belongs to the definition; later generic score() owns portfolio-wide
        # winsorization before z-score/rank/quintile construction.
        bounds = build.quantile([0.01, 0.99], axis=1).T
        build = build.clip(lower=bounds[0.01], upper=bounds[0.99], axis=0)
        return cash.rank(axis=1, pct=True, method="average").add(
            build.rank(axis=1, pct=True, method="average"), fill_value=np.nan
        ).sub(1.0)


RECIPES: dict[str, Recipe] = {
    "knowledge_capital_built_versus_bought": KnowledgeCapitalBuiltVersusBought(),
}


def get_recipe(name: str) -> Recipe:
    try:
        return RECIPES[name]
    except KeyError as error:
        known = ", ".join(sorted(RECIPES))
        raise KeyError(f"No independent PIT recipe named {name!r}. Available: {known}") from error
