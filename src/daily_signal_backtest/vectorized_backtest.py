from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .vectorized_weights import annual_weight_matrices


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


def _formation_dates(dates: pd.Series, frequency: str) -> pd.DatetimeIndex:
    if frequency == "daily":
        return pd.DatetimeIndex(dates.unique())
    period = "W-FRI" if frequency == "weekly" else "M"
    index = pd.DatetimeIndex(dates.unique())
    return pd.DatetimeIndex(pd.Series(index, index=index.to_period(period)).groupby(level=0).max())


def _metrics(frame: pd.DataFrame, nw_lags: int) -> dict:
    ret = frame.ls_return.dropna()
    wealth = (1 + ret).cumprod()
    model = sm.OLS(ret.to_numpy(), np.ones((len(ret), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    return {"n_days": len(ret), "AnnRet": wealth.iloc[-1] ** (252 / len(ret)) - 1, "AnnVol": ret.std(ddof=1) * np.sqrt(252), "Sharpe": ret.mean() / ret.std(ddof=1) * np.sqrt(252), "AvgDlyBets": frame.n_bets.mean(), "2WayTurnover": frame.two_way_turnover.mean(), "MDD": (wealth / wealth.cummax() - 1).min(), "t_stat": float(model.tvalues[0])}


def run_vectorized_grid(config: dict) -> None:
    """Compute every frequency and weight method from panel-level matrices."""
    root = Path(config["paths"]["derived_root"])
    raw_signals = json.loads((root / "signals" / "annual_signal_events_manifest.json").read_text())["signals"]
    signals = [LABELS.get(signal, signal) for signal in raw_signals]
    methods, frequencies = ("zscore", "rank", "quintile"), ("daily", "weekly", "monthly")
    cuts = config["universe"]["nyse_microcap_percentiles"]
    weight_cols = [f"{method}__{signal}" for method in methods for signal in signals]
    records = {(f, m, c): [] for f in frequencies for m in methods for c in cuts}
    history = {(frequency, cut): pd.DataFrame() for frequency in frequencies for cut in cuts}
    tolerance = config["portfolio"]["nonzero_weight_tolerance"]

    for year in range(2000, 2026):
        raw = pd.read_parquet(root / "daily_signals" / f"year={year}" / "data.parquet", columns=["date", "permno", "ret"])
        raw.date = pd.to_datetime(raw.date)
        forms = {frequency: _formation_dates(raw.date, frequency) for frequency in frequencies}
        for cut in cuts:
            formed = annual_weight_matrices(config, year, cut, raw_signals).drop(columns="ret")
            formed = formed.rename(columns={f"{method}__{old}": f"{method}__{new}" for method in methods for old, new in zip(raw_signals, signals)})
            base = raw.merge(formed, on=["date", "permno"], how="left")
            for frequency in frequencies:
                is_form = base.date.isin(forms[frequency])
                target = base[weight_cols].copy()
                target.loc[is_form] = target.loc[is_form].fillna(0.0)
                target.loc[~is_form] = np.nan
                prior = history[(frequency, cut)]
                work = pd.concat([prior, base[["date", "permno", "ret"]].join(target)], ignore_index=True).sort_values(["permno", "date"])
                work[weight_cols] = work.groupby("permno", sort=False)[weight_cols].ffill()
                held = work.groupby("permno", sort=False)[weight_cols].shift(2)
                work[weight_cols] = held
                current = work.loc[work.date.dt.year.eq(year)].copy()
                for method in methods:
                    columns = [f"{method}__{signal}" for signal in signals]
                    pnl = current[columns].mul(current.ret.fillna(0.0), axis=0).groupby(current.date).sum()
                    bets = (current[columns].abs() > tolerance).groupby(current.date).sum()
                    turnover = current[columns].sub(current.groupby("permno", sort=False)[columns].shift(1)).abs().groupby(current.date).sum()
                    for signal in signals:
                        column = f"{method}__{signal}"
                        result = pd.DataFrame({"date": pnl.index, "ls_return": pnl[column].to_numpy(), "n_bets": bets[column].reindex(pnl.index).to_numpy(), "two_way_turnover": turnover[column].reindex(pnl.index).fillna(0.0).to_numpy()})
                        result["microcap_percentile"], result["signal"] = cut, signal
                        records[(frequency, method, cut)].append(result)
                history[(frequency, cut)] = work.sort_values(["permno", "date"]).groupby("permno", sort=False).tail(2)[["date", "permno", "ret", *weight_cols]]
        print(f"vectorized grid processed {year}", flush=True)

    for frequency in frequencies:
        for method in methods:
            daily = pd.concat([part for cut in cuts for part in records[(frequency, method, cut)]], ignore_index=True).sort_values(["signal", "date"])
            daily["cum_return"] = daily.groupby(["microcap_percentile", "signal"])["ls_return"].transform(lambda x: (1 + x).cumprod() - 1)
            out = root / "backtest" / frequency / method
            out.mkdir(parents=True, exist_ok=True)
            daily.to_parquet(out / "daily_ls_returns.parquet", index=False)
            metrics = pd.DataFrame([{"microcap_percentile": cut, "signal": signal, **_metrics(group, config["portfolio"]["newey_west_lags"])} for (cut, signal), group in daily.groupby(["microcap_percentile", "signal"])])
            metrics.insert(0, "weight_method", method)
            metrics.insert(0, "rebalance_frequency", frequency)
            metrics.to_csv(out / "performance_metrics.csv", index=False)
            print(f"wrote {frequency}/{method}: {len(metrics)} portfolios", flush=True)
