from __future__ import annotations
import argparse
from pathlib import Path
from daily_signal_backtest.config import load_config
from daily_signal_backtest.audit import audit_annual
from daily_signal_backtest.signals import build_annual_signal_events
from daily_signal_backtest.universe import extract_daily_stock_base

def main() -> None:
    parser = argparse.ArgumentParser(description="Daily signal research pipeline")
    parser.add_argument("stage", choices=["audit", "signals", "universe", "daily-signals", "materialize-panels", "materialize-sic", "materialize-wide-panels", "materialize-wide-base", "materialize-wide-signals", "materialize-wide-nyse", "marketcap-panel", "panel-grid", "cache-weights", "wide-grid", "vectorized-daily", "vectorized-grid", "backtest", "backtest-grid", "report"])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--rebalance-frequency", choices=["daily", "weekly", "monthly"])
    parser.add_argument("--weight-method", choices=["zscore", "rank", "quintile"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.rebalance_frequency:
        config["portfolio"]["rebalance_frequency"] = args.rebalance_frequency
    if args.weight_method:
        config["portfolio"]["weight_method"] = args.weight_method
    root = Path(config["paths"]["derived_root"])
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "audit":
        audit_annual(config)
        return
    if args.stage == "signals":
        build_annual_signal_events(config)
        return
    if args.stage == "universe":
        extract_daily_stock_base(config)
        return
    if args.stage == "daily-signals":
        from daily_signal_backtest.daily_signals import materialize_daily_signals
        materialize_daily_signals(config)
        return
    if args.stage == "materialize-panels":
        from daily_signal_backtest.panel_store import materialize_value_panels
        materialize_value_panels(config)
        return
    if args.stage == "materialize-sic":
        from daily_signal_backtest.panel_store import materialize_sic_panel
        materialize_sic_panel(config)
        return
    if args.stage == "materialize-wide-panels":
        from daily_signal_backtest.wide_panels import materialize_wide_panels
        materialize_wide_panels(config)
        return
    if args.stage == "materialize-wide-base":
        from daily_signal_backtest.wide_panels import materialize_base_wide_panels
        materialize_base_wide_panels(config)
        return
    if args.stage == "materialize-wide-signals":
        from daily_signal_backtest.wide_panels import materialize_signal_wide_panels
        materialize_signal_wide_panels(config)
        return
    if args.stage == "materialize-wide-nyse":
        from daily_signal_backtest.wide_universe import materialize_nyse_panel
        materialize_nyse_panel(config)
        return
    if args.stage == "marketcap-panel":
        from daily_signal_backtest.marketcap_panel import build_marketcap_panels
        build_marketcap_panels(config)
        return
    if args.stage == "panel-grid":
        from daily_signal_backtest.panel_backtest import run_panel_grid
        run_panel_grid(config)
        return
    if args.stage == "wide-grid":
        from daily_signal_backtest.wide_backtest import run_wide_grid
        run_wide_grid(config)
        return
    if args.stage == "cache-weights":
        import json
        from daily_signal_backtest.wide_backtest import _build_weight_caches
        signals = config["grid"].get("signals") or json.loads((root / "signals" / "annual_signal_events_manifest.json").read_text())["signals"]
        _build_weight_caches(config, signals, config["grid"]["microcap_percentiles"], config["grid"]["sic_levels"])
        return
    if args.stage == "vectorized-daily":
        from daily_signal_backtest.vectorized_backtest import run_vectorized_daily
        run_vectorized_daily(config)
        return
    if args.stage == "vectorized-grid":
        from daily_signal_backtest.vectorized_backtest import run_vectorized_grid
        run_vectorized_grid(config)
        return
    if args.stage == "backtest":
        from daily_signal_backtest.backtest import run_backtest
        run_backtest(config)
        return
    if args.stage == "backtest-grid":
        from daily_signal_backtest.backtest import run_backtest_grid
        run_backtest_grid(config)
        return
    raise SystemExit(f"Stage '{args.stage}' is scaffolded. Derived root: {root}")

if __name__ == "__main__":
    main()
