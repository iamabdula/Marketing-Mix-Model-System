import unittest

import pandas as pd

from mmm.data_ingestion import DataIngestion
from mmm.utils.data_cleaner import clean_dataset
from mmm.utils.validators import merged_data_schema


class TestDataIngestion(unittest.TestCase):

    def setUp(self):
        self.ingestion = DataIngestion(data_dir="data")

    def test_merge_all_data(self):
        df = self.ingestion.merge_all_data()
        self.assertIn("date", df.columns)

        expected_cols = [
            "tv_spend",
            "radio_spend",
            "social_media_spend",
            "search_spend",
            "print_spend",
            "outdoor_spend",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_clean_dataset(self):
        data = {
            "date": pd.date_range("2022-01-01", periods=3),
            "sales": [100, -50, None],
            "tv_spend": [100, -10, None],
            "radio_spend": [50, 0, None],
            "social_media_spend": [20, -5, None],
            "search_spend": [10, 0, None],
            "print_spend": [5, 0, None],
            "outdoor_spend": [15, -3, None],
        }
        df = pd.DataFrame(data)
        cleaned_df = clean_dataset(df)

        self.assertEqual((cleaned_df["sales"] < 0).sum(), 0)
        self.assertEqual(cleaned_df["sales"].isna().sum(), 0)

        for col in [c for c in cleaned_df.columns if c.endswith("_spend")]:
            self.assertEqual((cleaned_df[col] < 0).sum(), 0)

    def test_final_schema_validation(self):
        df = self.ingestion.load_all_data()
        validated_df = merged_data_schema.validate(df)
        self.assertFalse(validated_df.empty)


if __name__ == "__main__":
    unittest.main()
