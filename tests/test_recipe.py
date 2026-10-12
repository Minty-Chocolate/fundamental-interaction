import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl

from us_equities_signal_research.model import FundamentalEvents
from us_equities_signal_research.recipes import _asof_event_panel, compute_recipes, get_recipe, signal_names


class _CCM:
    def map_events(self, events: pl.DataFrame) -> pl.DataFrame:
        return events.with_columns(
            pl.when(pl.col("gvkey") == "000001").then(pl.lit(1)).otherwise(pl.lit(2)).alias("permno")
        )


class RecipeTests(unittest.TestCase):
    def test_catalog_contains_all_eleven_published_recipes(self) -> None:
        self.assertEqual(len(signal_names()), 11)
        self.assertIn('knowledge_capital_built_versus_bought', signal_names())

    def test_industry_rank_and_zscore_match_paper_group_semantics(self) -> None:
        from us_equities_signal_research.recipes import _industry_transform

        dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
        values = pd.DataFrame([[1.0, 3.0, 5.0, 2.0, float("nan")], [1.0, 1.0, 4.0, 8.0, 2.0]], index=dates)
        sic = pd.DataFrame([[1001, 1001, 1001, 2001, 2001], [1001, 1001, 1001, 2001, 2001]], index=dates)
        rank = _industry_transform(values, sic, "rank")
        zscore = _industry_transform(values, sic, "zscore")
        self.assertEqual(rank.loc[dates[0]].iloc[:4].tolist(), [1 / 3, 2 / 3, 1.0, 1.0])
        self.assertEqual(rank.loc[dates[1]].iloc[:5].tolist(), [0.5, 0.5, 1.0, 1.0, 0.5])
        self.assertAlmostEqual(zscore.loc[dates[0], 0], -1.0)
        self.assertAlmostEqual(zscore.loc[dates[0], 1], 0.0)
        self.assertAlmostEqual(zscore.loc[dates[0], 2], 1.0)
        self.assertTrue(pd.isna(zscore.loc[dates[0], 3]))
        self.assertAlmostEqual(zscore.loc[dates[1], 0], -1 / np.sqrt(3))
        self.assertAlmostEqual(zscore.loc[dates[1], 2], 2 / np.sqrt(3))

    def test_industry_transform_matches_long_form_reference_on_mixed_groups(self) -> None:
        from us_equities_signal_research.recipes import _industry_transform

        rng = np.random.default_rng(7)
        dates = pd.date_range("2020-01-01", periods=7, freq="B")
        values = pd.DataFrame(rng.choice([float("nan"), -1.0, 0.0, 1.0, 2.0], size=(7, 9)), index=dates)
        sic = pd.DataFrame(rng.choice([1001, 1201, 2001, 4901, float("nan")], size=(7, 9)), index=dates)
        for method in ("rank", "zscore"):
            expected = pd.DataFrame(np.nan, index=dates, columns=values.columns)
            for date in dates:
                valid = values.loc[date].notna() & sic.loc[date].notna()
                grouped = values.loc[date, valid].groupby(np.floor(sic.loc[date, valid] / 100.0))
                if method == "rank":
                    expected.loc[date, valid] = grouped.rank(pct=True, method="average")
                else:
                    expected.loc[date, valid] = grouped.transform(lambda series: (series - series.mean()) / series.std())
            actual = _industry_transform(values, sic, method)
            np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True, rtol=1e-12, atol=1e-12)

    def test_asof_panel_clears_a_stale_value_after_null_revision(self) -> None:
        events = pd.DataFrame(
            {
                "gvkey": ["000001", "000001", "000001"],
                "available_date": pd.to_datetime(["2020-01-02", "2020-01-06", "2020-01-08"]),
                "value": [1.0, float("nan"), 3.0],
            }
        )
        calendar = pd.date_range("2020-01-02", "2020-01-08", freq="B")
        panel = _asof_event_panel(
            events, data=SimpleNamespace(ccm=_CCM()), calendar=calendar, permnos=(1,)
        )
        self.assertEqual(panel.loc[pd.Timestamp("2020-01-03"), 1], 1.0)
        self.assertTrue(pd.isna(panel.loc[pd.Timestamp("2020-01-06"), 1]))
        self.assertTrue(pd.isna(panel.loc[pd.Timestamp("2020-01-07"), 1]))
        self.assertEqual(panel.loc[pd.Timestamp("2020-01-08"), 1], 3.0)

    def test_recipe_asofs_events_without_lookahead(self) -> None:
        records = []
        for gvkey, cash, research in (("000001", 10.0, 5.0), ("000002", 5.0, 1.0)):
            for year in range(2015, 2020):
                records.append(
                    {
                        "gvkey": gvkey,
                        "fiscal_period_end": f"{year}-12-31",
                        "available_date": "2020-01-03",
                        "at": 100.0,
                        "oancf": cash,
                        "xrd": research,
                        "intan": 0.0,
                    }
                )
        events = FundamentalEvents(
            annual=pl.from_dicts(records),
            data=SimpleNamespace(ccm=_CCM()),
            coverage_permnos=(1, 2),
        )
        calendar = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
        panel = get_recipe('knowledge_capital_built_versus_bought').compute(events, calendar)
        self.assertTrue(panel.loc[pd.Timestamp("2020-01-02")].isna().all())
        self.assertTrue(panel.loc[pd.Timestamp("2020-01-03")].notna().all())
        self.assertGreater(panel.loc[pd.Timestamp("2020-01-03"), 1], panel.loc[pd.Timestamp("2020-01-03"), 2])

    def test_batched_recipes_match_individual_including_industry_inputs(self) -> None:
        records = []
        for gvkey, shift, sich in (("000001", 0.0, 2001.0), ("000002", 2.0, 6001.0)):
            for year in range(2015, 2020):
                records.append(
                    {
                        "gvkey": gvkey,
                        "fiscal_period_end": f"{year}-12-31",
                        "available_date": "2020-01-03",
                        "wcap": 5.0 + shift,
                        "che": 1.0,
                        "at": 100.0,
                        "revt": 30.0 + shift,
                        "cogs": 10.0,
                        "sich": sich,
                    }
                )
        events = FundamentalEvents(
            annual=pl.from_dicts(records),
            data=SimpleNamespace(ccm=_CCM()),
            coverage_permnos=(1, 2),
        )
        calendar = pd.date_range("2020-01-02", "2020-01-06", freq="B")
        recipes = (
            get_recipe("PredictableAccrualsSustainedProfit"),
            get_recipe("NegativeWorkingCapitalProfitSort"),
        )
        expected = {recipe.name: recipe.compute(events, calendar) for recipe in recipes}
        actual = compute_recipes(recipes, events, calendar)
        for name in expected:
            np.testing.assert_allclose(actual[name].to_numpy(), expected[name].to_numpy(), equal_nan=True)


if __name__ == "__main__":
    unittest.main()
