import unittest

import numpy as np
import pandas as pd

from us_equities_signal_research.backtest import excess_returns, rebalance_targets


class BacktestHelpersTests(unittest.TestCase):
    def test_weekly_targets_hold_last_formation_weights(self) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-12"])
        target = pd.DataFrame({1: range(7), 2: range(10, 17)}, index=dates, dtype=float)
        held = rebalance_targets(target, "weekly")
        self.assertTrue(held.iloc[:4].isna().all().all())
        self.assertEqual(held.loc[pd.Timestamp("2024-01-05")].tolist(), [4.0, 14.0])
        self.assertEqual(held.loc[pd.Timestamp("2024-01-08")].tolist(), [4.0, 14.0])

    def test_rebalance_closes_names_absent_from_new_formation_target(self) -> None:
        dates = pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
        ])
        target = pd.DataFrame(
            {1: [np.nan, np.nan, np.nan, np.nan, 0.5, np.nan, np.nan, np.nan, np.nan, np.nan],
             2: [np.nan, np.nan, np.nan, np.nan, -0.5, np.nan, np.nan, np.nan, np.nan, 0.25]},
            index=dates,
        )
        held = rebalance_targets(target, "weekly")
        self.assertEqual(held.loc[pd.Timestamp("2024-01-08")].tolist(), [0.5, -0.5])
        self.assertEqual(held.loc[pd.Timestamp("2024-01-12")].tolist(), [0.0, 0.25])

    def test_daily_rebalance_does_not_carry_forward_missing_weights(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        target = pd.DataFrame({1: [0.5, np.nan, 0.25]}, index=dates)
        held = rebalance_targets(target, "daily")
        self.assertTrue(np.isnan(held.iloc[1, 0]))

    def test_excess_return_uses_observable_raw_signal_cross_section(self) -> None:
        dates = pd.date_range("2024-01-02", periods=2, freq="B")
        returns = pd.DataFrame({1: [0.10, 0.03], 2: [0.02, 0.01], 3: [0.50, 0.04]}, index=dates)
        raw = pd.DataFrame({1: [1.0, 1.0], 2: [2.0, np.nan], 3: [np.nan, 3.0]}, index=dates)
        adjusted = excess_returns(returns, raw)
        self.assertAlmostEqual(adjusted.loc[dates[0], 1], 0.04)
        self.assertAlmostEqual(adjusted.loc[dates[0], 2], -0.04)
        self.assertAlmostEqual(adjusted.loc[dates[1], 1], -0.005)
        self.assertAlmostEqual(adjusted.loc[dates[1], 3], 0.005)


if __name__ == "__main__":
    unittest.main()
