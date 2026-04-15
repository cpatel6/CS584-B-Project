import os
import tempfile
import unittest

import pandas as pd

from src.evaluation.error_analysis import analyze_errors, categorize_error_text
from src.evaluation.robustness import PerturbationTesting
from src.utils.metrics import compute_metrics, measure_inference_speed


class ProjectRequirementTests(unittest.TestCase):
    def test_robustness_noise_is_deterministic(self):
        tester = PerturbationTesting(seed=7)
        text = "This movie was absolutely wonderful"
        self.assertEqual(tester.inject_noise(text, 0.2), tester.inject_noise(text, 0.2))

    def test_data_regime_scaling(self):
        tester = PerturbationTesting(seed=7)
        df = pd.DataFrame({"text": [str(i) for i in range(100)], "label": [i % 2 for i in range(100)]})
        scaled = tester.scale_data_regime(df, 0.1)
        self.assertEqual(len(scaled), 10)

    def test_error_taxonomy_categories(self):
        self.assertEqual(categorize_error_text("I did not like this"), "negation")
        self.assertEqual(categorize_error_text("yeah right this was great"), "sarcasm")
        self.assertEqual(categorize_error_text("fast shipping but bad quality"), "domain_specific_words")
        self.assertEqual(categorize_error_text("soooo goood"), "typos")

    def test_error_analysis_outputs_taxonomy_file(self):
        y_true = [0, 1]
        y_pred = [1, 0]
        texts = ["not good", "yeah right amazing"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, _, errors = analyze_errors(y_true, y_pred, texts, "test_model", tmp_dir)
            self.assertIn("category", errors.columns)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "test_model_error_taxonomy.csv")))

    def test_metrics_include_speed(self):
        y_true = [0, 1, 1]
        y_pred = [0, 1, 0]
        metrics = compute_metrics(y_true, y_pred)
        self.assertIn("f1_macro", metrics)
        speed = measure_inference_speed(lambda x: [0 for _ in x], [1, 2, 3], repeats=2)
        self.assertGreater(speed["inference_samples_per_sec"], 0.0)


if __name__ == "__main__":
    unittest.main()
