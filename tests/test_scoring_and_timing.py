import unittest
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

from us_equities_signal_research import RawSignalPanel, realized_returns, score
from us_equities_signal_research.execution import DenseReturnEngine
from us_equities_signal_research.execution import return_observations


@dataclass(frozen=True)
class _Rule:
    start: date = date(2024, 1, 3)


@dataclass(frozen=True)
class _Universe:
    membership: pl.DataFrame
    rule: _Rule = _Rule()
    report_start: date = date(2024, 1, 3)


class ScoringAndTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.date_range("2024-01-01", periods=5, freq="B")
        self.universe = _Universe(
            pl.DataFrame(
                {
                    "date": self.dates,
                    "1": [True] * 5,
                    "2": [True] * 5,
                    "3": [True] * 5,
                }
            )
        )

    def test_score_keeps_missing_signals_missing_and_normalizes_globally(self) -> None:
        raw = RawSignalPanel(
            name="test",
            universe=self.universe,
            values=pd.DataFrame(
                [[1.0, 2.0, np.nan], [1.0, 3.0, 5.0], [2.0, 2.0, 2.0], [1.0, 2.0, 3.0], [5.0, 2.0, 1.0]],
                index=self.dates,
                columns=[1, 2, 3],
            ),
        )
        weights = score(raw, method="zscore", winsorize=0.0).values
        self.assertTrue(np.isnan(weights.loc[self.dates[0], 3]))
        self.assertAlmostEqual(weights.loc[self.dates[1]].sum(), 0.0, places=12)
        self.assertAlmostEqual(weights.loc[self.dates[1]].abs().sum(), 1.0, places=12)
        self.assertEqual(weights.loc[self.dates[2]].tolist(), [0.0, 0.0, 0.0])

    def test_realized_return_uses_two_session_shift(self) -> None:
        weights = pd.DataFrame(
            {1: [1.0, 0.0, 0.0, 0.0, 0.0], 2: [-1.0, 0.0, 0.0, 0.0, 0.0]},
            index=self.dates,
        )
        returns = pd.DataFrame(
            {1: [0.0, 0.0, 0.10, 0.0, 0.0], 2: [0.0, 0.0, -0.05, 0.0, 0.0]},
            index=self.dates,
        )
        pnl = realized_returns(weights, returns, report_start=self.dates[0])
        self.assertTrue(np.isnan(pnl.iloc[0]))
        self.assertTrue(np.isnan(pnl.iloc[1]))
        self.assertAlmostEqual(pnl.iloc[2], 0.15)

    def test_sparse_observation_index_matches_dense_pnl(self) -> None:
        weights = pd.DataFrame(
            {1: [0.5, 0.4, np.nan, 0.3, 0.2], 2: [-0.5, -0.4, np.nan, -0.3, -0.2], 3: [np.nan] * 5},
            index=self.dates,
        )
        returns = pd.DataFrame(
            {1: [0.01, np.nan, 0.03, 0.02, 0.01], 2: [-0.01, -0.02, -0.03, np.nan, -0.01], 3: [np.nan] * 5},
            index=self.dates,
        )
        dense = realized_returns(weights, returns, report_start=self.dates[0])
        sparse = realized_returns(
            weights, returns, report_start=self.dates[0], observations=return_observations(returns)
        )
        np.testing.assert_allclose(sparse.to_numpy(), dense.to_numpy(), equal_nan=True)

    def test_reusable_dense_engine_matches_reference_pnl(self) -> None:
        weights = pd.DataFrame(
            {1: [0.5, 0.4, np.nan, 0.3, 0.2], 2: [-0.5, -0.4, np.nan, -0.3, -0.2]},
            index=self.dates,
        )
        returns = pd.DataFrame(
            {1: [0.01, np.nan, 0.03, 0.02, 0.01], 2: [-0.01, -0.02, -0.03, np.nan, -0.01]},
            index=self.dates,
        )
        expected = realized_returns(weights, returns, report_start=self.dates[0])
        actual = DenseReturnEngine(returns, report_start=self.dates[0]).realized(weights)
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)

    def test_grouped_scores_match_pandas_reference_for_all_methods(self) -> None:
        from us_equities_signal_research.scoring import prepare_score, score_prepared_many

        values = pd.DataFrame(
            [[1.0, 1.0, 3.0], [2.0, np.nan, 2.0], [1.0, 2.0, 4.0]],
            index=self.dates[:3], columns=[1, 2, 3],
        )
        groups = pd.DataFrame(
            [[1.0, 1.0, 2.0], [1.0, 2.0, 2.0], [1.0, np.nan, 1.0]],
            index=self.dates[:3], columns=[1, 2, 3],
        )
        raw = RawSignalPanel(name="test", universe=self.universe, values=values)
        actual = score_prepared_many(prepare_score(raw, winsorize=0.0), groups=groups, normalize=None)
        for method, scored in actual.items():
            expected = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
            for day in values.index:
                valid = values.loc[day].notna() & groups.loc[day].notna()
                grouped = values.loc[day, valid].groupby(groups.loc[day, valid])
                if method == "zscore":
                    expected.loc[day, valid] = grouped.transform(
                        lambda series: (series - series.mean()) / series.std(ddof=0)
                    ).fillna(0.0)
                elif method == "rank":
                    expected.loc[day, valid] = grouped.rank(pct=True, method="average") * 2.0 - 1.0
                else:
                    percentile = grouped.rank(pct=True, method="average")
                    expected.loc[day, valid] = np.select(
                        [percentile.le(0.2), percentile.ge(0.8)], [-1.0, 1.0], default=0.0
                    )
            np.testing.assert_allclose(scored.values.to_numpy(), expected.to_numpy(), equal_nan=True)


if __name__ == "__main__":
    unittest.main()

class PerformanceTests(unittest.TestCase):
    def test_metrics_use_held_two_session_weights(self) -> None:
        from us_equities_signal_research import performance_metrics

        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        target = pd.DataFrame({1: [1.0, 0.0, 0.0, 0.0], 2: [-1.0, 0.0, 0.0, 0.0]}, index=dates)
        pnl = pd.Series([np.nan, np.nan, 0.02, 0.01], index=dates)
        metrics = performance_metrics(pnl, target)
        self.assertEqual(metrics.loc[0, "n_days"], 2)
        self.assertAlmostEqual(metrics.loc[0, "AvgDlyBets"], 1.0)
