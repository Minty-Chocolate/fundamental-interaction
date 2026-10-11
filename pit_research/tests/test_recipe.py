import unittest
from types import SimpleNamespace

import pandas as pd
import polars as pl

from us_equities_signal_research.model import FundamentalEvents
from us_equities_signal_research.recipes import KnowledgeCapitalBuiltVersusBought


class _CCM:
    def map_events(self, events: pl.DataFrame) -> pl.DataFrame:
        return events.with_columns(
            pl.when(pl.col("gvkey") == "000001").then(pl.lit(1)).otherwise(pl.lit(2)).alias("permno")
        )


class RecipeTests(unittest.TestCase):
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
        panel = KnowledgeCapitalBuiltVersusBought().compute(events, calendar)
        self.assertTrue(panel.loc[pd.Timestamp("2020-01-02")].isna().all())
        self.assertTrue(panel.loc[pd.Timestamp("2020-01-03")].notna().all())
        self.assertGreater(panel.loc[pd.Timestamp("2020-01-03"), 1], panel.loc[pd.Timestamp("2020-01-03"), 2])


if __name__ == "__main__":
    unittest.main()
