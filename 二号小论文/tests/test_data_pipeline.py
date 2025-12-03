# tests/test_data_pipeline.py

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.your_dl_project.data_processing import preprocess_data
from src.your_dl_project.dataset import CustomDataset


class TestDataPipeline(unittest.TestCase):
    """针对数据预处理与数据集封装的基础单元测试。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, "toy_dataset.csv")
        self.raw_df = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, np.nan],
                "feature_2": [0.1, 0.2, 0.3],
                "label": [0, 1, 1],
            }
        )
        self.raw_df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preprocess_data_removes_missing_rows(self):
        processed = preprocess_data(self.raw_df)
        expected_rows = len(self.raw_df.dropna())
        self.assertEqual(len(processed), expected_rows)
        self.assertFalse(processed.isna().any().any())

    def test_custom_dataset_len_and_transform(self):
        transform = lambda sample: sample.values.astype(np.float32)
        dataset = CustomDataset(self.csv_path, transform=transform)

        self.assertEqual(len(dataset), len(self.raw_df))

        transformed_sample = dataset[0]
        self.assertIsInstance(transformed_sample, np.ndarray)
        self.assertEqual(transformed_sample.shape[0], len(self.raw_df.columns))


if __name__ == "__main__":
    unittest.main()
