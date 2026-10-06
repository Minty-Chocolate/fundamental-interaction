from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def annual_weight_matrices(config: dict, year: int, cutoff: float, signals: list[str]) -> pd.DataFrame:
    """Compute all 11 daily cross-sectional weight matrices without date loops."""
    root = Path(config["paths"]["derived_root"])
    cap = pd.read_parquet(root / "panels" / "market_cap" / f"year={year}.parquet")
    masks = pd.read_parquet(root / "panels" / "market_cap" / f"universe_year={year}.parquet")
    daily = pd.read_parquet(root / "daily_signals" / f"year={year}" / "data.parquet", columns=["date", "permno", "ret", *signals])
    cap.index = pd.to_datetime(cap.index)
    masks["permno"] = masks["permno"].astype(str)
    nyse = masks.loc[masks.is_nyse, ["date", "permno"]].assign(value=1).pivot(index="date", columns="permno", values="value")
    nyse = nyse.reindex(index=cap.index, columns=cap.columns).notna()
    threshold = cap.where(nyse).quantile(cutoff, axis=1)
    daily["permno_key"] = daily["permno"].astype(str)
    cap_long = cap.stack(future_stack=True).rename("lagged_market_cap").reset_index()
    cap_long.columns = ["date", "permno_key", "lagged_market_cap"]
    daily = daily.merge(cap_long, on=["date", "permno_key"], how="left")
    masks["permno_int"] = pd.to_numeric(masks["permno"], errors="raise").astype("int64")
    daily = daily.merge(masks[["date", "permno_int", "is_nonfinancial", "has_return"]], left_on=["date", "permno"], right_on=["date", "permno_int"], how="left").drop(columns="permno_int")
    daily["threshold"] = daily["date"].map(threshold)
    base = daily.loc[daily.is_nonfinancial & daily.has_return & daily.lagged_market_cap.gt(daily.threshold)].copy()
    values = base[signals].apply(pd.to_numeric, errors="coerce")
    grouped = values.groupby(base["date"])
    lo = grouped.quantile(config["portfolio"]["winsor_lower"]).reindex(base["date"]).set_axis(base.index)
    hi = grouped.quantile(config["portfolio"]["winsor_upper"]).reindex(base["date"]).set_axis(base.index)
    clipped = values.clip(lower=lo, upper=hi, axis="index")
    z = clipped.sub(clipped.groupby(base["date"]).transform("mean")).div(clipped.groupby(base["date"]).transform("std", ddof=0))
    rank = clipped.groupby(base["date"]).rank(pct=True, method="average")
    output = base[["date", "permno", "ret"]].copy()
    for method, score in {"zscore": z, "rank": rank.sub(.5)}.items():
        weights = score.div(score.abs().groupby(base["date"]).transform("sum"))
        output = output.join(weights.add_prefix(f"{method}__"))
    q = pd.DataFrame(np.nan, index=rank.index, columns=rank.columns).mask(rank.notna(), 0.0)
    q = q.mask(rank.ge(.8), .5 / rank.ge(.8).groupby(base["date"]).transform("sum"))
    q = q.mask(rank.le(.2), -.5 / rank.le(.2).groupby(base["date"]).transform("sum"))
    return output.join(q.add_prefix("quintile__"))
