"""Normalize existing backtest result labels without running a backtest."""

from pathlib import Path

import pandas as pd


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


def normalize(root: Path) -> None:
    for path in root.glob("**/performance_metrics.csv"):
        frame = pd.read_csv(path)
        frame["signal"] = frame["signal"].replace(LABELS)
        frame.to_csv(path, index=False)
        print(path)
    for path in root.glob("**/daily_ls_returns.parquet"):
        frame = pd.read_parquet(path)
        frame["signal"] = frame["signal"].replace(LABELS)
        frame.to_parquet(path, index=False)
        print(path)


if __name__ == "__main__":
    normalize(Path("data/derived/backtest"))
