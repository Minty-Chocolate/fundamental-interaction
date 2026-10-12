"""Independent PIT implementations of the paper's eleven published signals.

The formula strings are retained only as compact, auditable definitions.  This
module has its own parser and evaluator; it never imports the reference
``SignalEngine`` or the retired backtest package.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import numpy as np
import pandas as pd
import polars as pl

from .model import FundamentalEvents, Recipe


@dataclass(frozen=True)
class _Node:
    op: str
    args: tuple["_Node", ...] = ()
    value: str | float | None = None


_CROSS_SECTIONAL = frozenset({"RANK", "WINSOR", "SUMRANK", "IND_RANK", "IND_ZSCORE"})
_TEMPORAL = frozenset({"DELTA", "GROWTH", "MA", "VOL", "TS_SUM", "TS_MIN", "TS_MAX"})

# Frozen from paper/data/signal_catalog.json.  It is deliberately copied here
# instead of importing the reference repository at runtime.
_FORMULAS: tuple[tuple[str, str], ...] = (
    (
        "purchased_share_of_growth_spending",
        "SUB(RANK(MA(RATIO(oancf, at), 3)), RANK(RATIO(TS_SUM(COALESCE(aqc, SUB(at, at)), 5), ADD(TS_SUM(COALESCE(aqc, SUB(at, at)), 5), TS_SUM(COALESCE(xrd, SUB(at, at)), 5)))))",
    ),
    (
        "knowledge_capital_built_versus_bought",
        "SUMRANK(MA(RATIO(oancf, at), 3), WINSOR(SUB(RATIO(TS_SUM(COALESCE(xrd, SUB(at, at)), 5), at), RATIO(COALESCE(intan, SUB(at, at)), at)), 1))",
    ),
    (
        "PredictableAccrualsSustainedProfit",
        "MUL(IND_RANK(NEG(WINSOR(VOL(RATIO(DELTA(SUB(wcap, che), 1), at), 5), 1))), IND_RANK(WINSOR(MA(RATIO(SUB(revt, cogs), at), 3), 1)))",
    ),
    (
        "Funded Research Tilt With Profit Delivery",
        "MUL(RANK(NEG(RATIO(SUB(DELTA(capx, 1), DELTA(COALESCE(xrd, 0), 1)), MA(sale, 3)))), RANK(GROWTH(SUB(sale, cogs), 1)))",
    ),
    (
        "WorkingCapitalBloatVsPeers",
        "SUMRANK(NEG(IND_ZSCORE(RATIO(ADD(SUB(wcap, che), dlc), sale))), IND_ZSCORE(RATIO(SUB(revt, cogs), at)))",
    ),
    (
        "ConservatismAdjustedProfitLeanClaims",
        "MIN(IND_RANK(RATIO(ADD(oiadp, COALESCE(xrd, 0)), at)), IND_RANK(NEG(RATIO(SUB(act, che), sale))))",
    ),
    (
        "Sustained Coverage With Capped Cushion",
        "WINSOR(MA(RATIO(ADD(SUB(revt, cogs), MIN(ADD(xsga, COALESCE(xrd, 0)), ADD(ADD(xint, txt), dlc))), ADD(ADD(xint, txt), dlc)), 3), 1)",
    ),
    (
        "Broad Universe Profitable Value Model",
        "SUMRANK(RATIO(ADD(SUB(revt, ADD(cogs, COALESCE(xsga, 0))), COALESCE(xrd, 0)), at), SUMRANK(NEG(RATIO(SUB(ib, COALESCE(oancf, ADD(ib, dp))), at)), SUMRANK(RATIO(ADD(ceq, TS_SUM(COALESCE(xrd, 0), 3)), MUL(prcc_f, csho)), NEG(COALESCE(RATIO(SUB(ADD(dltt, dlc), che), oibdp), RATIO(SUB(ADD(dltt, dlc), che), at))))))",
    ),
    (
        "NegativeWorkingCapitalProfitSort",
        "MUL(INDICATOR(NEG(wcap), 0), IND_ZSCORE(RATIO(SUB(revt, cogs), at)))",
    ),
    (
        "TroughCoverageCleanAccrualRollover",
        "MIN(IND_RANK(TS_MIN(RATIO(SUB(oancf, dp), lt), 5)), MIN(IND_RANK(RATIO(SUB(che, dlc), at)), IND_RANK(NEG(RATIO(SUB(pi, oancf), revt)))))",
    ),
    (
        "Gross Margin Inventory Efficiency",
        "MUL(IND_RANK(RATIO(SUB(revt, cogs), sale)), IND_RANK(NEG(RATIO(invt, sale))))",
    ),
)


def _split_arguments(text: str) -> list[str]:
    pieces: list[str] = []
    depth, start = 0, 0
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            pieces.append(text[start:index].strip())
            start = index + 1
    pieces.append(text[start:].strip())
    return pieces


def _parse(text: str) -> _Node:
    text = text.strip()
    try:
        return _Node("NUMBER", value=float(text))
    except ValueError:
        pass
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$", text, re.DOTALL)
    if match:
        name, body = match.groups()
        return _Node(name.upper(), tuple(_parse(part) for part in _split_arguments(body)))
    return _Node("VAR", value=text.lower())


def _field_names(node: _Node) -> set[str]:
    if node.op == "VAR":
        return {str(node.value)}
    names: set[str] = set()
    for child in node.args:
        names.update(_field_names(child))
    return names


def _contains_cross_section(node: _Node) -> bool:
    return node.op in _CROSS_SECTIONAL or any(_contains_cross_section(child) for child in node.args)


def _uses_industry(node: _Node) -> bool:
    return node.op.startswith("IND_") or any(_uses_industry(child) for child in node.args)


def _parameter(node: _Node, default: int | float) -> int | float:
    if node.op != "NUMBER" or node.value is None:
        raise ValueError("Formula parameter must be a numeric literal.")
    return node.value


def _last_non_null(values: pd.Series) -> float:
    observed = pd.to_numeric(values, errors="coerce").dropna().unique()
    if len(observed) == 1:
        return float(observed[0])
    # Simultaneous conflicting revisions cannot be ordered at daily resolution.
    return float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= 1e-10:
        return float("nan")
    return numerator / denominator


def _nan_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _nan_sum(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.sum(finite)) if finite else float("nan")


def _nan_min(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.min(finite)) if finite else float("nan")


def _nan_max(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.max(finite)) if finite else float("nan")


def _eval_scalar(node: _Node, history: dict[int, dict[str, float]], year: int) -> float:
    """Evaluate an expression for one firm at one observable accounting state.

    ``history`` is reconstructed as arrivals are observed.  Temporal operators
    use exact fiscal-calendar years, matching the paper code's year-based lag
    behavior: a missing intervening year never borrows an adjacent statement.
    Cross-sectional operators are intentionally excluded and are applied only
    after these scalar PIT states are mapped to daily panels.
    """
    if node.op == "NUMBER":
        return float(node.value)
    if node.op == "VAR":
        return float(history.get(year, {}).get(str(node.value), np.nan))
    if node.op in _CROSS_SECTIONAL:
        raise ValueError(f"{node.op} requires a cross-sectional daily panel.")

    values = lambda: [_eval_scalar(child, history, year) for child in node.args]
    if node.op == "NEG":
        return -_eval_scalar(node.args[0], history, year)
    if node.op == "ADD":
        a, b = values()
        return a + b if np.isfinite(a) and np.isfinite(b) else float("nan")
    if node.op == "SUB":
        a, b = values()
        return a - b if np.isfinite(a) and np.isfinite(b) else float("nan")
    if node.op == "MUL":
        a, b = values()
        return a * b if np.isfinite(a) and np.isfinite(b) else float("nan")
    if node.op == "RATIO":
        return _safe_ratio(*values())
    if node.op == "COALESCE":
        a, b = values()
        return a if np.isfinite(a) else b
    if node.op == "MIN":
        a, b = values()
        return min(a, b) if np.isfinite(a) and np.isfinite(b) else float("nan")
    if node.op == "MAX":
        a, b = values()
        return max(a, b) if np.isfinite(a) and np.isfinite(b) else float("nan")
    if node.op == "INDICATOR":
        value = _eval_scalar(node.args[0], history, year)
        threshold = float(_parameter(node.args[1], 0.0)) if len(node.args) > 1 else 0.0
        return float(value > threshold) if np.isfinite(value) else float("nan")

    if node.op in _TEMPORAL:
        expression = node.args[0]
        window = int(_parameter(node.args[1], 1)) if len(node.args) > 1 else 1
        current = _eval_scalar(expression, history, year)
        if node.op == "DELTA":
            lagged = _eval_scalar(expression, history, year - window)
            return current - lagged if np.isfinite(current) and np.isfinite(lagged) else float("nan")
        if node.op == "GROWTH":
            lagged = _eval_scalar(expression, history, year - window)
            return _safe_ratio(current - lagged, abs(lagged))
        history_values = [_eval_scalar(expression, history, year - lag) for lag in range(window)]
        if node.op == "MA":
            return _nan_mean(history_values)
        if node.op == "VOL":
            finite = [value for value in history_values if np.isfinite(value)]
            return float(np.std(finite, ddof=1)) if len(finite) >= 2 else float("nan")
        if node.op == "TS_SUM":
            return _nan_sum(history_values)
        if node.op == "TS_MIN":
            return _nan_min(history_values)
        if node.op == "TS_MAX":
            return _nan_max(history_values)
    raise ValueError(f"Unsupported formula operator: {node.op}")


def _scalar_events_many(annual: pd.DataFrame, nodes: tuple[_Node, ...], fields: tuple[str, ...]) -> dict[_Node, pd.DataFrame]:
    """Evaluate several scalar formula roots while reconstructing firm state once."""
    use = ["gvkey", "available_date", "fiscal_period_end", *fields]
    work = annual.loc[:, use].copy()
    work["gvkey"] = work["gvkey"].astype(str).str.zfill(6)
    work["available_date"] = pd.to_datetime(work["available_date"])
    work["fiscal_period_end"] = pd.to_datetime(work["fiscal_period_end"])
    work = work.dropna(subset=["gvkey", "available_date", "fiscal_period_end"])
    for field in fields:
        work[field] = pd.to_numeric(work[field], errors="coerce")
    work = work.sort_values(["gvkey", "available_date", "fiscal_period_end"])

    records: dict[_Node, list[dict[str, object]]] = {node: [] for node in nodes}
    for gvkey, firm in work.groupby("gvkey", sort=False):
        state: dict[int, dict[str, float]] = {}
        period_ends: dict[int, pd.Timestamp] = {}
        for available_date, arrivals in firm.groupby("available_date", sort=False):
            for fiscal_end, revision_rows in arrivals.groupby("fiscal_period_end", sort=False):
                fiscal_end = pd.Timestamp(fiscal_end)
                fiscal_year = fiscal_end.year
                existing_end = period_ends.get(fiscal_year)
                if existing_end is not None and existing_end != fiscal_end:
                    state[fiscal_year] = {}
                    period_ends[fiscal_year] = fiscal_end
                    continue
                period_ends[fiscal_year] = fiscal_end
                destination = state.setdefault(fiscal_year, {})
                for field in fields:
                    value = _last_non_null(revision_rows[field])
                    if np.isfinite(value):
                        destination[field] = value
            if not period_ends:
                continue
            latest_year = max(period_ends, key=lambda key: period_ends[key])
            for node in nodes:
                records[node].append(
                    {
                        "gvkey": gvkey,
                        "available_date": pd.Timestamp(available_date),
                        "value": _eval_scalar(node, state, latest_year),
                    }
                )
    return {
        node: pd.DataFrame.from_records(values, columns=["gvkey", "available_date", "value"])
        for node, values in records.items()
    }


def _scalar_roots(node: _Node) -> tuple[_Node, ...]:
    """Maximal scalar subexpressions consumed by daily cross-sectional ops."""
    if not _contains_cross_section(node):
        return (node,)
    if node.op in {"RANK", "WINSOR"}:
        return _scalar_roots(node.args[0])
    if node.op in {"IND_RANK", "IND_ZSCORE"}:
        return _scalar_roots(node.args[0]) + (_Node("VAR", value="sich"),)
    if node.op == "SUMRANK":
        return _scalar_roots(node.args[0]) + _scalar_roots(node.args[1])
    roots: tuple[_Node, ...] = ()
    for child in node.args:
        roots += _scalar_roots(child)
    return roots


def _asof_event_panel(
    events: pd.DataFrame,
    *,
    data: object,
    calendar: pd.DatetimeIndex,
    permnos: tuple[int, ...],
) -> pd.DataFrame:
    """Map sparse observable states through CCM and carry them on a calendar.

    A null value on an event date is a *new observable state*, not an absent
    event.  The explicit presence matrix below prevents ``ffill`` from
    incorrectly retaining a stale signal after a revision makes it undefined.
    """
    base = pd.DataFrame(index=calendar, columns=pd.Index(permnos, dtype="int64"), dtype="float64")
    if events.empty:
        return base
    mapped = data.ccm.map_events(pl.from_pandas(events)).to_pandas()
    if mapped.empty:
        return base
    mapped["available_date"] = pd.to_datetime(mapped["available_date"])
    mapped["permno"] = pd.to_numeric(mapped["permno"], errors="coerce").astype("Int64")
    mapped = mapped.loc[mapped["permno"].isin(permnos)].dropna(subset=["available_date", "permno"])
    if mapped.empty:
        return base
    mapped["permno"] = mapped["permno"].astype("int64")
    collapsed = (
        mapped.groupby(["available_date", "permno"], sort=True)["value"]
        .agg(_last_non_null)
        .rename("value")
        .reset_index()
    )
    presence = mapped.groupby(["available_date", "permno"], sort=True).size().rename("present").reset_index()
    dates = pd.DatetimeIndex(sorted(set(calendar).union(pd.DatetimeIndex(collapsed["available_date"]))), name="date")
    values = collapsed.pivot(index="available_date", columns="permno", values="value").reindex(index=dates, columns=base.columns)
    seen = (
        presence.pivot(index="available_date", columns="permno", values="present")
        .reindex(index=dates, columns=base.columns)
        .fillna(0)
        .astype(bool)
    )
    value_array = values.to_numpy(dtype=float, na_value=np.nan)
    seen_array = seen.to_numpy(dtype=bool)
    state_array = np.full(value_array.shape, np.nan, dtype=float)
    row = np.arange(len(dates), dtype=np.int64)[:, None]
    # Every event, including an event whose value is null, advances the state.
    # Work in column blocks to avoid a Python loop per security while keeping
    # temporary index matrices bounded for the top-2,000 universe.
    block_width = 256
    for start in range(0, value_array.shape[1], block_width):
        stop = min(start + block_width, value_array.shape[1])
        observed = seen_array[:, start:stop]
        latest_event = np.where(observed, row, -1)
        latest_event = np.maximum.accumulate(latest_event, axis=0)
        active = latest_event >= 0
        if not active.any():
            continue
        width = stop - start
        columns = np.broadcast_to(np.arange(width, dtype=np.int64), latest_event.shape)
        block = state_array[:, start:stop]
        block[active] = value_array[:, start:stop][latest_event[active], columns[active]]
        state_array[:, start:stop] = block
    state = pd.DataFrame(state_array, index=dates, columns=base.columns)
    return state.reindex(calendar)


def _daily_rank(values: pd.DataFrame) -> pd.DataFrame:
    return values.rank(axis=1, pct=True, method="average")


def _daily_winsor(values: pd.DataFrame, percent: float) -> pd.DataFrame:
    probability = percent / 100.0
    bounds = values.quantile([probability, 1.0 - probability], axis=1).T
    return values.clip(lower=bounds[probability], upper=bounds[1.0 - probability], axis=0)


def _industry_rank_chunk(observed: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Average-tie percentile ranks for contiguous ``(date, SIC)`` groups."""
    order = np.lexsort((observed, key))
    ordered_values = observed[order]
    ordered_key = key[order]
    size = len(order)
    positions = np.arange(size, dtype=np.int64)
    group_start = np.r_[True, ordered_key[1:] != ordered_key[:-1]]
    group_end = np.r_[ordered_key[:-1] != ordered_key[1:], True]
    start_position = np.maximum.accumulate(np.where(group_start, positions, -1))
    end_position = np.minimum.accumulate(np.where(group_end, positions, size)[::-1])[::-1]
    within_group = positions - start_position + 1
    group_size = end_position - start_position + 1
    tie_start = group_start | np.r_[True, ordered_values[1:] != ordered_values[:-1]]
    tie_end = group_end | np.r_[ordered_values[:-1] != ordered_values[1:], True]
    tie_start_position = np.maximum.accumulate(np.where(tie_start, positions, -1))
    tie_end_position = np.minimum.accumulate(np.where(tie_end, positions, size)[::-1])[::-1]
    average_rank = (within_group[tie_start_position] + within_group[tie_end_position]) / 2.0
    ranked = np.empty(size, dtype=float)
    ranked[order] = average_rank / group_size
    return ranked


def _industry_transform(values: pd.DataFrame, sic: pd.DataFrame, method: str) -> pd.DataFrame:
    """Paper ``IND_*`` behavior: daily, within two-digit Compustat SIC.

    Computation is batched in a bounded number of calendar rows.  It avoids a
    multi-million-row pandas long table and also avoids thousands of Python
    date-level group-bys, while retaining pandas' average-tie rank and sample
    standard-deviation definitions.
    """
    value_array = values.to_numpy(dtype=float, na_value=np.nan)
    sic_array = sic.to_numpy(dtype=float, na_value=np.nan)
    transformed = np.full(value_array.shape, np.nan, dtype=float)
    n_rows, n_columns = value_array.shape
    row_chunk = 256
    key_base = 10_000
    for first in range(0, n_rows, row_chunk):
        last = min(first + row_chunk, n_rows)
        current = value_array[first:last]
        current_sic = sic_array[first:last]
        valid = np.isfinite(current) & np.isfinite(current_sic)
        if not valid.any():
            continue
        flat_positions = np.flatnonzero(valid)
        observed = current.ravel()[flat_positions]
        code = np.floor(current_sic.ravel()[flat_positions] / 100.0).astype(np.int64)
        day = flat_positions // n_columns
        key = day * key_base + code
        _, inverse = np.unique(key, return_inverse=True)
        if method == "zscore":
            count = np.bincount(inverse)
            total = np.bincount(inverse, weights=observed)
            mean = total / count
            deviation = observed - mean[inverse]
            squared = np.bincount(inverse, weights=deviation * deviation)
            std = np.full(len(count), np.nan, dtype=float)
            multiple = count > 1
            std[multiple] = np.sqrt(squared[multiple] / (count[multiple] - 1))
            std[std <= 1e-10] = np.nan
            output = deviation / std[inverse]
        else:
            output = _industry_rank_chunk(observed, key)
        block = transformed[first:last]
        block.ravel()[flat_positions] = output
        transformed[first:last] = block
    return pd.DataFrame(transformed, index=values.index, columns=values.columns)


@dataclass
class _EvaluationContext:
    events: FundamentalEvents
    calendar: pd.DatetimeIndex
    fields: tuple[str, ...]
    annual: pd.DataFrame
    scalar_panels: dict[_Node, pd.DataFrame]
    daily_panels: dict[_Node, pd.DataFrame]
    industry: pd.DataFrame | None = None

    @property
    def permnos(self) -> tuple[int, ...]:
        return self.events.coverage_permnos


def _evaluate_daily(node: _Node, context: _EvaluationContext) -> pd.DataFrame:
    cached = context.daily_panels.get(node)
    if cached is not None:
        return cached
    if not _contains_cross_section(node):
        cached = context.scalar_panels.get(node)
        if cached is None:
            sparse = _scalar_events(context.annual, node, context.fields)
            cached = _asof_event_panel(sparse, data=context.events.data, calendar=context.calendar, permnos=context.permnos)
            context.scalar_panels[node] = cached
        context.daily_panels[node] = cached
        return cached

    if node.op == "RANK":
        result = _daily_rank(_evaluate_daily(node.args[0], context))
    elif node.op == "WINSOR":
        result = _daily_winsor(_evaluate_daily(node.args[0], context), float(_parameter(node.args[1], 1.0)))
    elif node.op == "SUMRANK":
        result = _daily_rank(_evaluate_daily(node.args[0], context)).add(
            _daily_rank(_evaluate_daily(node.args[1], context)), fill_value=np.nan
        ).sub(1.0)
    elif node.op in {"IND_RANK", "IND_ZSCORE"}:
        if context.industry is None:
            industry_node = _Node("VAR", value="sich")
            context.industry = _evaluate_daily(industry_node, context)
        method = "rank" if node.op == "IND_RANK" else "zscore"
        result = _industry_transform(_evaluate_daily(node.args[0], context), context.industry, method)
    elif node.op == "NEG":
        result = -_evaluate_daily(node.args[0], context)
    elif node.op in {"ADD", "SUB", "MUL", "RATIO", "MIN", "MAX", "COALESCE"}:
        left, right = (_evaluate_daily(child, context) for child in node.args)
        if node.op == "ADD":
            result = left + right
        elif node.op == "SUB":
            result = left - right
        elif node.op == "MUL":
            result = left * right
        elif node.op == "RATIO":
            result = left / right.where(right.abs() > 1e-10)
        elif node.op == "MIN":
            result = pd.DataFrame(np.minimum(left.to_numpy(), right.to_numpy()), index=left.index, columns=left.columns)
        elif node.op == "MAX":
            result = pd.DataFrame(np.maximum(left.to_numpy(), right.to_numpy()), index=left.index, columns=left.columns)
        else:
            result = left.where(left.notna(), right)
    elif node.op == "INDICATOR":
        value = _evaluate_daily(node.args[0], context)
        threshold = float(_parameter(node.args[1], 0.0)) if len(node.args) > 1 else 0.0
        result = (value > threshold).astype(float).where(value.notna())
    else:
        raise ValueError(f"Cannot assemble daily formula operator {node.op}.")
    context.daily_panels[node] = result
    return result


@dataclass(frozen=True)
class PublishedSignalRecipe:
    """One immutable published formula with a single executable method."""

    name: str
    expression: str
    required_fields: tuple[str, ...]
    annual_lookback_years: int = 5
    prefetch_buffer_years: int = 2
    version: str = "pit_daily_v2"

    def compute(self, events: FundamentalEvents, calendar: pd.DatetimeIndex) -> pd.DataFrame:
        expression = _parse(self.expression)
        annual = events.annual
        columns = pd.Index(events.coverage_permnos, dtype="int64")
        if annual.is_empty():
            return pd.DataFrame(index=calendar, columns=columns, dtype="float64")
        annual_frame = events.annual_pandas if events.annual_pandas is not None else annual.to_pandas()
        roots = tuple(dict.fromkeys(_scalar_roots(expression)))
        sparse_by_root = _scalar_events_many(annual_frame, roots, self.required_fields)
        return _evaluate_from_sparse(
            expression, self.required_fields, events, calendar, annual_frame, sparse_by_root,
        ).reindex(index=calendar, columns=columns)


def _evaluate_from_sparse(
    expression: _Node,
    fields: tuple[str, ...],
    events: FundamentalEvents,
    calendar: pd.DatetimeIndex,
    annual_frame: pd.DataFrame,
    sparse_by_root: dict[_Node, pd.DataFrame],
) -> pd.DataFrame:
    """Apply daily cross-sectional operators to precomputed scalar events."""
    roots = tuple(dict.fromkeys(_scalar_roots(expression)))
    context = _EvaluationContext(
        events=events,
        calendar=calendar,
        fields=fields,
        annual=annual_frame,
        scalar_panels={
            root: _asof_event_panel(
                sparse_by_root[root], data=events.data, calendar=calendar, permnos=events.coverage_permnos
            )
            for root in roots
        },
        daily_panels={},
    )
    return _evaluate_daily(expression, context)


def compute_recipes(
    recipes: tuple[PublishedSignalRecipe, ...],
    events: FundamentalEvents,
    calendar: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Compute several recipes with one per-firm Annual-state reconstruction."""
    if not recipes:
        return {}
    annual = events.annual
    columns = pd.Index(events.coverage_permnos, dtype="int64")
    if annual.is_empty():
        return {recipe.name: pd.DataFrame(index=calendar, columns=columns, dtype="float64") for recipe in recipes}
    annual_frame = events.annual_pandas if events.annual_pandas is not None else annual.to_pandas()
    expressions = {recipe.name: _parse(recipe.expression) for recipe in recipes}
    fields = tuple(sorted({field for recipe in recipes for field in recipe.required_fields}))
    roots = tuple(dict.fromkeys(
        root for expression in expressions.values() for root in _scalar_roots(expression)
    ))
    sparse_by_root = _scalar_events_many(annual_frame, roots, fields)
    return {
        recipe.name: _evaluate_from_sparse(
            expressions[recipe.name], recipe.required_fields, events, calendar, annual_frame, sparse_by_root,
        ).reindex(index=calendar, columns=columns)
        for recipe in recipes
    }


def _recipe(name: str, expression: str) -> PublishedSignalRecipe:
    parsed = _parse(expression)
    fields = _field_names(parsed)
    if _uses_industry(parsed):
        fields.add("sich")
    return PublishedSignalRecipe(name=name, expression=expression, required_fields=tuple(sorted(fields)))


RECIPES: dict[str, Recipe] = {name: _recipe(name, expression) for name, expression in _FORMULAS}


def get_recipe(name: str) -> Recipe:
    try:
        return RECIPES[name]
    except KeyError as error:
        known = ", ".join(RECIPES)
        raise KeyError(f"No PIT recipe named {name!r}. Available: {known}") from error


def signal_names() -> tuple[str, ...]:
    return tuple(RECIPES)
