# US Equities Alpha Research

A daily, point-in-time research workflow for the eleven published signals from
*Can AI Do Financial Research? LLM-Guided Hypothesis Discovery in Asset
Pricing*. It uses local CRSP daily returns, Compustat North America Annual, and
the time-valid CCM link history through `us_equities_datahandler`.

> **Data availability.** The repository contains code, notebooks, tests, and
> presentation figures only. CRSP, Compustat, and CCM are licensed WRDS/vendor
> datasets and are deliberately excluded. Point the data-handler configuration
> at a locally licensed copy before running the pipeline.

The active implementation is `us_equities_signal_research`. The former
`daily_signal_backtest` runtime lives under [`retired/legacy_runtime`](retired/legacy_runtime)
and is not imported or executed.

## What it does

- Reconstructs each firm's Annual accounting history only from records whose
  `available_date` has arrived (`fdate`, otherwise `datadate + 6 months`).
- Applies time-valid CCM links and produces a cached daily `date × permno` raw
  panel for every signal.
- Implements the eleven paper formulas independently, including formula-level
  `RANK`, `WINSOR`, `SUMRANK`, and two-digit-Compustat-SIC industry operators.
- Forms equal-risk long–short portfolios with global 1% winsorization,
  `zscore`, `rank`, or `quintile` weights; optional SIC-division scoring; and
  daily, weekly, or monthly rebalancing.  Raw panels and fully scored daily
  weights are cached separately on the fast local volume.
- Uses the strict timing convention: a signal observed after close on `t`
  forms at `t+1` close and earns the `t+1 → t+2` close-to-close return.

The standard run evaluates three dynamically reconstituted CRSP universes by
lagged market capitalization: top 500, top 1,000, and top 2,000, from 2000
through 2025. Daily stock returns
are made excess by subtracting the equal-weight return of securities with a
non-null raw signal on that date.

## Completed daily portfolio grid

The checked-in figures use daily rebalancing, global scoring, and z-score
weights. The first figure is the top-500 baseline; the second overlays the
three dynamic market-cap universes for each of the eleven signals.

![Top-500 daily z-score portfolios](assets/top500_daily_zscore_global.png)

![Universe sensitivity](assets/topn_daily_zscore_global.png)

## Run

```bash
docker compose build
docker compose run --rm research python -m us_equities_signal_research.cli --rank-ltes 500,1000,2000
```

The result cache is written to the fast local `CACHE_ROOT`, not T7:

```text
/cache/backtests/
  top500_2000_2025/
  top1000_2000_2025/
  top2000_2000_2025/
```

Open Jupyter Lab with `docker compose up -d`, then use
[`notebooks/01_pit_backtest_report.ipynb`](notebooks/01_pit_backtest_report.ipynb)
to run or inspect the complete grid. The grid parameters are ordinary Python
lists, so the report can overlay any Cartesian product of frequency, weight
method, and SIC treatment for all eleven signals.

## Repository layout

```text
src/us_equities_signal_research/  PIT signal recipes, scoring, and backtests
notebooks/                        reproducible research report
tests/                            timing, formula, and portfolio tests
assets/                           committed figures shown above
retired/legacy_runtime/           frozen provenance archive; never execute
```

Generated data, result caches, notebook checkpoints, and local configuration
are excluded by `.gitignore`. This keeps the repository portable while the
large licensed inputs and derived panels remain local.
