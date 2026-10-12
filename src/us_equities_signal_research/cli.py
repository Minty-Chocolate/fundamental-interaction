"""Command-line entry point for the full PIT signal backtest grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import us_equities_datahandler as dh

from .backtest import FREQUENCIES, SIC_LEVELS, WEIGHT_METHODS, run
from .recipes import signal_names


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _sic_csv(value: str) -> tuple[int | None, ...]:
    result: list[int | None] = []
    for item in _csv(value):
        if item.lower() in {"none", "global"}:
            result.append(None)
        elif item == "1":
            result.append(1)
        else:
            raise argparse.ArgumentTypeError("sic levels must be global/none or 1")
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent PIT daily US-equity signal backtests.")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--rank-lte", type=int, default=500, help="One daily lagged-market-cap rank universe.")
    parser.add_argument("--rank-ltes", default=None, help="Comma-separated rank universes, for example 500,1000,2000.")
    parser.add_argument("--output", default=None, help="Output directory for one rank universe; incompatible with --rank-ltes.")
    parser.add_argument("--output-root", default="/cache/backtests", help="Parent directory used with --rank-ltes.")
    parser.add_argument("--signals", default=",".join(signal_names()))
    parser.add_argument("--frequencies", default=",".join(FREQUENCIES))
    parser.add_argument("--methods", default=",".join(WEIGHT_METHODS))
    parser.add_argument("--sic-levels", default="global,1")
    parser.add_argument("--refresh-raw", action="store_true")
    parser.add_argument("--refresh-results", action="store_true", help="Rebuild a matching completed result grid.")
    parser.add_argument("--crsp-root", default="/datahandler/data/derived/crsp_ranked_daily")
    parser.add_argument("--annual", default="/datahandler/data/derived/compustat/annual.parquet")
    parser.add_argument("--ccm", default="/datahandler/data/derived/ccm/links.parquet")
    args = parser.parse_args()

    ranks = (args.rank_lte,) if args.rank_ltes is None else tuple(int(value) for value in _csv(args.rank_ltes))
    if not ranks or any(rank < 1 for rank in ranks):
        raise ValueError("rank universes must be positive integers")
    if args.output is not None and len(ranks) != 1:
        parser.error("--output is only valid for one universe; use --output-root with --rank-ltes")

    dh.configure(crsp_ranked_daily=args.crsp_root, compustat_annual=args.annual, ccm_links=args.ccm)
    for rank in ranks:
        universe = dh.universe.market_cap(start=args.start, end=args.end, rank_lte=rank, lag=1)
        output = (
            Path(args.output)
            if args.output is not None
            else Path(args.output_root) / f"top{rank}_{args.start[:4]}_{args.end[:4]}"
        )
        daily, metrics = run(
            data=dh,
            universe=universe,
            output=output,
            names=_csv(args.signals),
            frequencies=_csv(args.frequencies),
            methods=_csv(args.methods),
            sic_levels=_sic_csv(args.sic_levels),
            refresh_raw=args.refresh_raw,
            refresh_results=args.refresh_results,
        )
        print(f"top {rank:,}: wrote {len(daily):,} daily rows and {len(metrics):,} metric rows to {output}")


if __name__ == "__main__":
    main()
