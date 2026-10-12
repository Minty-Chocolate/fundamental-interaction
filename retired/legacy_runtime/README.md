# Retired legacy runtime

**Status: retired. Do not use this implementation for research, backtests, or
new development.**

This directory preserves the original `daily_signal_backtest` code as a
historical reference. Its data assembly, signal availability, portfolio
construction, and result cache are superseded by the independent point-in-time
pipeline in [`src/us_equities_signal_research`](../../src/us_equities_signal_research).

The active project entry points are:

- [`README.md`](../../README.md) for the current research design and commands;
- [`notebooks/00_pit_daily_signal.ipynb`](../../notebooks/00_pit_daily_signal.ipynb)
  to build daily PIT signal panels; and
- [`notebooks/01_pit_backtest_report.ipynb`](../../notebooks/01_pit_backtest_report.ipynb)
  to inspect backtests and figures.

The Docker, configuration, notebooks, and source files retained here are
intentionally frozen. They may encode obsolete assumptions and must not be
treated as a reproducible version of the current results. No active package,
test, Docker image, or documented command imports this directory.
