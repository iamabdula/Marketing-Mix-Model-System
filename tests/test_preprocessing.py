import unittest

import numpy as np
import pandas as pd

from mmm.preprocessing import Preprocessor, apply_adstock, apply_saturation


class TestPreprocessing(unittest.TestCase):
    def test_apply_adstock(self):
        spend = np.array([100, 0, 0])
        decay = 0.5
        result = apply_adstock(spend, decay).tolist()
        self.assertEqual(result, [100, 50, 25])

    def test_apply_saturation(self):
        spend = np.array([0, 50, 100])
        alpha, gamma = 1.0, 1.0
        result = apply_saturation(spend, alpha, gamma)

        # Ensure monotonic increase
        self.assertTrue(all(x <= y for x, y in zip(result, result[1:])))

    def test_preprocessor_transform(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=3),
                "tv_spend": [100, 0, 50],
                "radio_spend": [0, 30, 10],
            }
        )

        preprocessor = Preprocessor(
            decay_factors={"tv_spend": 0.5, "radio_spend": 0.3},
            alpha_params={"tv_spend": 1.0, "radio_spend": 1.0},
            gamma_params={"tv_spend": 1.0, "radio_spend": 1.0},
        )

        transformed_df = preprocessor.transform(df, ["tv_spend", "radio_spend"])

        self.assertIn("tv_spend_transformed", transformed_df.columns)
        self.assertIn("radio_spend_transformed", transformed_df.columns)
        self.assertTrue(transformed_df["tv_spend_transformed"].notna().all())
        self.assertTrue(transformed_df["radio_spend_transformed"].notna().all())


if __name__ == "__main__":
    unittest.main()
