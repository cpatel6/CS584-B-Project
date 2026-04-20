import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.evaluation.error_analysis import analyze_errors, categorize_error_text
from src.evaluation.robustness import PerturbationTesting
from src.utils.metrics import compute_metrics, measure_inference_speed
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_metrics_bar,
    plot_domain_shift_drop_heatmap,
    plot_all_metrics_heatmap,
    plot_robustness_summary,
    plot_error_taxonomy_crossmodel,
    plot_shuffle_robustness,
)
from src.models.statistical import NGramSVM


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
        speed = measure_inference_speed(lambda inputs: [0 for _ in inputs], [1, 2, 3], repeats=2)
        self.assertGreater(speed["inference_samples_per_sec"], 0.0)

    def test_ngram_svm_feature_importance(self):
        """NGramSVM must expose get_feature_importance after fitting."""
        svm = NGramSVM(max_features=50)
        texts = ["good product", "bad product", "great", "terrible", "excellent"]
        labels = [1, 0, 1, 0, 1]
        svm.fit(texts, labels)
        names, coefs = svm.get_feature_importance()
        self.assertEqual(len(names), len(coefs))
        self.assertGreater(len(names), 0)

    def test_domain_shift_returns_tuple(self):
        """evaluate_domain_shift must return (dict, preds_in, preds_out)."""
        from src.evaluation.domain_shift import evaluate_domain_shift
        from src.models.statistical import TFIDFLR

        model = TFIDFLR(max_features=20)
        texts_train = ["good film", "bad movie", "love it", "hate it", "great", "terrible"]
        labels_train = [1, 0, 1, 0, 1, 0]
        model.fit(texts_train, labels_train)

        df_in = pd.DataFrame({"text": ["great film", "bad movie"], "label": [1, 0]})
        df_out = pd.DataFrame({"text": ["love it", "hate it"], "label": [1, 0]})
        result, p_in, p_out = evaluate_domain_shift(model, df_in, df_out, "TFIDF_LR")

        self.assertIn("in_domain_accuracy", result)
        self.assertIn("in_domain_precision", result)
        self.assertEqual(len(p_in), len(df_in))
        self.assertEqual(len(p_out), len(df_out))

    # ── Plot smoke tests ────────────────────────────────────────────────────

    def _shift_df(self):
        models = ["A", "B", "C"]
        return pd.DataFrame({
            "model": models,
            "in_domain_accuracy":  [0.9, 0.85, 0.8],
            "out_domain_accuracy": [0.82, 0.76, 0.70],
            "accuracy_drop":       [0.08, 0.09, 0.10],
            "in_domain_f1":        [0.9, 0.85, 0.8],
            "out_domain_f1":       [0.82, 0.76, 0.70],
            "f1_drop":             [0.08, 0.09, 0.10],
            "in_domain_precision": [0.9, 0.85, 0.8],
            "in_domain_recall":    [0.9, 0.85, 0.8],
            "out_domain_precision":[0.82, 0.76, 0.70],
            "out_domain_recall":   [0.82, 0.76, 0.70],
        })

    def test_confusion_matrix_plot_saves_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cm.png")
            plot_confusion_matrix([0, 1, 0, 1], [0, 1, 1, 0],
                                  model_name="M", domain="In", save_path=path)
            self.assertTrue(os.path.exists(path))

    def test_metrics_bar_saves_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bar.png")
            plot_metrics_bar(self._shift_df(), metric="accuracy", save_path=path)
            self.assertTrue(os.path.exists(path))

    def test_drop_heatmap_saves_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "hm.png")
            plot_domain_shift_drop_heatmap(self._shift_df(), save_path=path)
            self.assertTrue(os.path.exists(path))

    def test_all_metrics_heatmap_saves_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "all.png")
            plot_all_metrics_heatmap(self._shift_df(), save_path=path)
            self.assertTrue(os.path.exists(path))

    def test_robustness_summary_saves_file(self):
        df = pd.DataFrame({
            "model": ["A", "B"] * 3,
            "noise_level": [0.05, 0.05, 0.1, 0.1, 0.15, 0.15],
            "accuracy": [0.8, 0.75, 0.7, 0.65, 0.6, 0.55],
            "f1_macro": [0.8, 0.75, 0.7, 0.65, 0.6, 0.55],
        })
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rob.png")
            plot_robustness_summary(df, save_path=path)
            self.assertTrue(os.path.exists(path))

    def test_error_taxonomy_crossmodel_saves_file(self):
        errors = pd.DataFrame({"category": ["negation", "typos", "other"]})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tax.png")
            plot_error_taxonomy_crossmodel({"M1": errors, "M2": errors}, save_path=path)
            self.assertTrue(os.path.exists(path))

    def test_shuffle_robustness_saves_file(self):
        df = pd.DataFrame({
            "model": ["A", "B"],
            "baseline_accuracy": [0.9, 0.85],
            "shuffled_accuracy": [0.55, 0.52],
        })
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "shuf.png")
            plot_shuffle_robustness(df, save_path=path)
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
