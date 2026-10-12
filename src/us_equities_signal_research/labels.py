"""Stable public labels for the paper's eleven signal recipes."""

from __future__ import annotations


_CANONICAL: dict[str, str] = {
    "purchased_share_of_growth_spending": "purchased_share_of_growth_spending",
    "knowledge_capital_built_versus_bought": "knowledge_capital_built_versus_bought",
    "PredictableAccrualsSustainedProfit": "predictable_accruals_sustained_profit",
    "Funded Research Tilt With Profit Delivery": "funded_research_tilt_with_profit_delivery",
    "WorkingCapitalBloatVsPeers": "working_capital_bloat_vs_peers",
    "ConservatismAdjustedProfitLeanClaims": "conservatism_adjusted_profit_lean_claims",
    "Sustained Coverage With Capped Cushion": "sustained_coverage_with_capped_cushion",
    "Broad Universe Profitable Value Model": "broad_universe_profitable_value_model",
    "NegativeWorkingCapitalProfitSort": "negative_working_capital_profit_sort",
    "TroughCoverageCleanAccrualRollover": "trough_coverage_clean_accrual_rollover",
    "Gross Margin Inventory Efficiency": "gross_margin_inventory_efficiency",
}


def canonical_signal_name(name: str) -> str:
    """Return the uniform lower-snake-case public label for a recipe name."""
    return _CANONICAL.get(name, name)
