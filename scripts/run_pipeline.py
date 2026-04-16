import yaml
import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.seed import seed_everything
from src.utils.logger import get_logger
from src.utils.metrics import measure_training_time, measure_inference_speed
from src.data.dataset_loader import DatasetLoader
from src.data.preprocessor import Preprocessor
from src.models.statistical import TFIDFLR, NGramSVM
from src.evaluation.metrics import compute_metrics
from src.visualization.plots import (
    plot_feature_importance,
    plot_degradation,
    plot_confusion_matrix,
    plot_metrics_bar,
    plot_domain_shift_drop_heatmap,
    plot_all_metrics_heatmap,
    plot_timing_comparison,
    plot_robustness_accuracy,
    plot_data_regime_accuracy,
    plot_error_taxonomy_crossmodel,
    plot_label_distribution,
    plot_generalization_scatter,
    plot_radar_comparison,
    plot_robustness_summary,
    plot_shuffle_robustness,
    plot_precision_recall_f1,
    plot_results_table,
    # ── new combined plots ──────────────────────────────────────────
    plot_accuracy_f1_combined,
    plot_precision_recall_f1_combined,
    plot_all_confusion_matrices,
    plot_feature_importance_combined,
    plot_data_regime_combined,
    plot_robustness_combined,
)
from src.evaluation.domain_shift import evaluate_domain_shift
from src.evaluation.robustness import PerturbationTesting
from src.evaluation.error_analysis import analyze_errors

logger = get_logger("MainPipeline")


def ensure_dirs(config):
    for d in [config['paths']['outputs_dir'], config['paths']['models_dir'],
              config['paths']['results_dir'], config['paths']['figures_dir'],
              config['paths']['reports_dir']]:
        os.makedirs(d, exist_ok=True)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


from src.models.embeddings import GloVeLR
from src.models.cnn import CNNText, NeuralTextClassifier
from src.models.bilstm import BiLSTMText
from src.models.distilbert import DistilBERTClassifier


def _save_model_checkpoint(model, model_name, models_dir):
    """Save model checkpoint with pickle first, then HF-style fallback when available."""
    os.makedirs(models_dir, exist_ok=True)
    try:
        path = os.path.join(models_dir, f"{model_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        return path
    except Exception as exc:
        logger.warning(f"Pickle checkpoint failed for {model_name}: {exc}")
        if hasattr(model, "model") and hasattr(model.model, "save_pretrained"):
            path = os.path.join(models_dir, model_name)
            model.model.save_pretrained(path)
            if hasattr(model, "tokenizer"):
                model.tokenizer.save_pretrained(path)
            return path
        raise


def run_pipeline_all_models(config, train_df, test_in_df, test_out_df, figs_dir):
    all_shift_results = []
    trained_models = {}
    preds_in_all = {}
    preds_out_all = {}
    # Collect feature-importance data for a combined plot at the end
    _feat_importance_pairs = []

    # ── STATISTICAL MODELS ──────────────────────────────────────────────────

    logger.info("--- Training Statistical Model 1: TF-IDF + LR ---")
    tfidf_lr = TFIDFLR(
        max_features=config['preprocessing']['max_features_tfidf'],
        random_state=config['seed'])
    train_time = measure_training_time(
        tfidf_lr.fit,
        train_df['text'].tolist(), train_df['label'].tolist())
    trained_models['TFIDF_LR'] = tfidf_lr
    res1, p_in1, p_out1 = evaluate_domain_shift(
        tfidf_lr, test_in_df, test_out_df, model_name="TFIDF_LR")
    preds_in_all['TFIDF_LR'] = p_in1
    preds_out_all['TFIDF_LR'] = p_out1
    inference_metrics = measure_inference_speed(
        tfidf_lr.predict, test_in_df['text'].tolist())
    res1.update(inference_metrics)
    res1["training_time_sec"] = train_time
    all_shift_results.append(res1)
    _save_model_checkpoint(tfidf_lr, "TFIDF_LR", config['paths']['models_dir'])

    features, coefs = tfidf_lr.get_feature_importance()
    _feat_importance_pairs.append(("TF-IDF + LR – Top Features", features, coefs))

    logger.info("--- Training Statistical Model 2: N-Grams + SVM ---")
    ngram_svm = NGramSVM(
        max_features=config['preprocessing']['max_features_tfidf'],
        ngram_range=(1, 2), random_state=config['seed'])
    train_time = measure_training_time(
        ngram_svm.fit,
        train_df['text'].tolist(), train_df['label'].tolist())
    trained_models['NGram_SVM'] = ngram_svm
    res2, p_in2, p_out2 = evaluate_domain_shift(
        ngram_svm, test_in_df, test_out_df, model_name="NGram_SVM")
    preds_in_all['NGram_SVM'] = p_in2
    preds_out_all['NGram_SVM'] = p_out2
    inference_metrics = measure_inference_speed(
        ngram_svm.predict, test_in_df['text'].tolist())
    res2.update(inference_metrics)
    res2["training_time_sec"] = train_time
    all_shift_results.append(res2)
    _save_model_checkpoint(ngram_svm, "NGram_SVM", config['paths']['models_dir'])

    ng_features, ng_coefs = ngram_svm.get_feature_importance()
    _feat_importance_pairs.append(("N-Gram + SVM – Top Features", ng_features, ng_coefs))

    logger.info("--- Training Statistical Model 3: GloVe + LR ---")
    glove_lr = GloVeLR(glove_path=None, random_state=config['seed'])
    train_time = measure_training_time(
        glove_lr.fit,
        train_df['text'].tolist(), train_df['label'].tolist())
    trained_models['GloVe_LR'] = glove_lr
    res3, p_in3, p_out3 = evaluate_domain_shift(
        glove_lr, test_in_df, test_out_df, model_name="GloVe_LR")
    preds_in_all['GloVe_LR'] = p_in3
    preds_out_all['GloVe_LR'] = p_out3
    inference_metrics = measure_inference_speed(
        glove_lr.predict, test_in_df['text'].tolist())
    res3.update(inference_metrics)
    res3["training_time_sec"] = train_time
    all_shift_results.append(res3)
    _save_model_checkpoint(glove_lr, "GloVe_LR", config['paths']['models_dir'])

    # ── NEURAL MODELS ───────────────────────────────────────────────────────

    logger.info("--- Training Neural Model 1: CNNText ---")
    cnn_model = NeuralTextClassifier(
        CNNText, glove_path=None,
        epochs=config['training']['epochs'],
        device=config['training']['device'])
    train_time = measure_training_time(
        cnn_model.fit,
        train_df['text'].tolist(), train_df['label'].tolist())
    trained_models['CNNText_GloVe'] = cnn_model
    res4, p_in4, p_out4 = evaluate_domain_shift(
        cnn_model, test_in_df, test_out_df, model_name="CNNText_GloVe")
    preds_in_all['CNNText_GloVe'] = p_in4
    preds_out_all['CNNText_GloVe'] = p_out4
    inference_metrics = measure_inference_speed(
        cnn_model.predict, test_in_df['text'].tolist())
    res4.update(inference_metrics)
    res4["training_time_sec"] = train_time
    all_shift_results.append(res4)
    _save_model_checkpoint(cnn_model, "CNNText_GloVe", config['paths']['models_dir'])

    logger.info("--- Training Neural Model 2: BiLSTM ---")
    bilstm_model = NeuralTextClassifier(
        BiLSTMText, glove_path=None, hidden_size=64,
        epochs=config['training']['epochs'],
        device=config['training']['device'])
    train_time = measure_training_time(
        bilstm_model.fit,
        train_df['text'].tolist(), train_df['label'].tolist())
    trained_models['BiLSTM_GloVe'] = bilstm_model
    res5, p_in5, p_out5 = evaluate_domain_shift(
        bilstm_model, test_in_df, test_out_df, model_name="BiLSTM_GloVe")
    preds_in_all['BiLSTM_GloVe'] = p_in5
    preds_out_all['BiLSTM_GloVe'] = p_out5
    inference_metrics = measure_inference_speed(
        bilstm_model.predict, test_in_df['text'].tolist())
    res5.update(inference_metrics)
    res5["training_time_sec"] = train_time
    all_shift_results.append(res5)
    _save_model_checkpoint(bilstm_model, "BiLSTM_GloVe",
                           config['paths']['models_dir'])

    logger.info("--- Training Neural Model 3: DistilBERT ---")
    distilbert_model = DistilBERTClassifier(device=config['training']['device'])
    train_time = measure_training_time(
        distilbert_model.fit,
        train_df['text'].tolist(), train_df['label'].tolist(),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
    )
    trained_models['DistilBERT'] = distilbert_model
    res6, p_in6, p_out6 = evaluate_domain_shift(
        distilbert_model, test_in_df, test_out_df, model_name="DistilBERT")
    preds_in_all['DistilBERT'] = p_in6
    preds_out_all['DistilBERT'] = p_out6
    inference_metrics = measure_inference_speed(
        distilbert_model.predict, test_in_df['text'].tolist())
    res6.update(inference_metrics)
    res6["training_time_sec"] = train_time
    all_shift_results.append(res6)
    _save_model_checkpoint(distilbert_model, "DistilBERT",
                           config['paths']['models_dir'])

    return all_shift_results, trained_models, preds_in_all, preds_out_all, \
           _feat_importance_pairs


def main():
    config = load_config()
    ensure_dirs(config)
    seed_everything(config['seed'])

    figs_dir    = config['paths']['figures_dir']
    results_dir = config['paths']['results_dir']
    reports_dir = config['paths']['reports_dir']

    # ── Phase 1: Data ────────────────────────────────────────────────────────
    logger.info("Phase 1: Dataset Pipeline & Preprocessing")
    loader = DatasetLoader(config)
    train_df, val_df, test_in_df = loader.load_in_domain()
    test_out_df = loader.load_out_of_domain()

    preprocessor = Preprocessor(config)
    for df in [train_df, test_in_df, test_out_df]:
        df['text'] = df['text'].fillna('')

    train_df    = preprocessor.process_dataframe(train_df)
    test_in_df  = preprocessor.process_dataframe(test_in_df)
    test_out_df = preprocessor.process_dataframe(test_out_df)

    # Dataset statistics plot
    plot_label_distribution(
        {"Train (Amazon)": train_df,
         "Test In-Domain (Amazon)": test_in_df,
         "Test Out-of-Domain (IMDb)": test_out_df},
        save_path=os.path.join(figs_dir, 'label_distribution.png'))

    # ── Phase 2-5: Train all models ─────────────────────────────────────────
    (domain_shift_results, models, preds_in_all, preds_out_all,
     feat_importance_pairs) = run_pipeline_all_models(
        config, train_df, test_in_df, test_out_df, figs_dir)

    # ── Confusion matrices: combined grid + per-model individual files ────────
    logger.info("--- Generating confusion matrices ---")
    y_in_true  = test_in_df['label'].tolist()
    y_out_true = test_out_df['label'].tolist()

    # NEW: one grid with all models × both domains
    plot_all_confusion_matrices(
        y_in_true, y_out_true, preds_in_all, preds_out_all,
        save_path=os.path.join(figs_dir, 'all_confusion_matrices.png'))

    # Keep individual files for backward compatibility
    for name in models:
        plot_confusion_matrix(
            y_in_true, preds_in_all[name],
            model_name=name, domain="In-Domain (Amazon)",
            save_path=os.path.join(figs_dir,
                                   f'{name}_confusion_matrix_in_domain.png'))
        plot_confusion_matrix(
            y_out_true, preds_out_all[name],
            model_name=name, domain="Out-of-Domain (IMDb)",
            save_path=os.path.join(figs_dir,
                                   f'{name}_confusion_matrix_out_domain.png'))

    # ── Domain shift summary CSVs and plots ─────────────────────────────────
    df_shift = pd.DataFrame(domain_shift_results)
    df_shift.to_csv(os.path.join(results_dir, 'domain_shift_summary.csv'),
                    index=False)
    logger.info(f"\nFinal Comparative Domain Shift Metrics:\n{df_shift.to_string()}")

    timing_cols = ["model", "training_time_sec", "inference_time_sec",
                   "inference_samples_per_sec", "num_predictions"]
    available_timing_cols = [c for c in timing_cols if c in df_shift.columns]
    if available_timing_cols:
        df_timing = df_shift[available_timing_cols].copy()
        df_timing.to_csv(os.path.join(results_dir, 'timing_summary.csv'),
                         index=False)

    # In-domain metrics table (precision, recall, F1)
    in_domain_metric_cols = ["model", "in_domain_accuracy",
                              "in_domain_precision", "in_domain_recall",
                              "in_domain_f1"]
    avail_in = [c for c in in_domain_metric_cols if c in df_shift.columns]
    df_in_metrics = df_shift[avail_in].rename(
        columns={c: c.replace("in_domain_", "") for c in avail_in})
    df_in_metrics.to_csv(os.path.join(results_dir, 'in_domain_metrics.csv'),
                         index=False)

    out_domain_metric_cols = ["model", "out_domain_accuracy",
                               "out_domain_precision", "out_domain_recall",
                               "out_domain_f1"]
    avail_out = [c for c in out_domain_metric_cols if c in df_shift.columns]
    df_out_metrics = df_shift[avail_out].rename(
        columns={c: c.replace("out_domain_", "") for c in avail_out})
    df_out_metrics.to_csv(os.path.join(results_dir, 'out_domain_metrics.csv'),
                          index=False)

    # ── COMPARISON PLOTS ─────────────────────────────────────────────────────
    logger.info("--- Generating comprehensive comparison plots ---")

    # NEW: accuracy + F1 combined in one figure
    plot_accuracy_f1_combined(
        df_shift,
        save_path=os.path.join(figs_dir, 'accuracy_f1_combined.png'))

    # NEW: P/R/F1 for both domains in one figure
    plot_precision_recall_f1_combined(
        df_shift,
        save_path=os.path.join(figs_dir, 'precision_recall_f1_combined.png'))

    # NEW: combined feature importance (both statistical models side-by-side)
    if feat_importance_pairs:
        plot_feature_importance_combined(
            feat_importance_pairs,
            save_path=os.path.join(figs_dir, 'feature_importance_combined.png'))

    plot_domain_shift_drop_heatmap(
        df_shift,
        save_path=os.path.join(figs_dir, 'domain_shift_drop_heatmap.png'))
    plot_all_metrics_heatmap(
        df_shift,
        save_path=os.path.join(figs_dir, 'all_metrics_heatmap.png'))
    plot_generalization_scatter(
        df_shift,
        save_path=os.path.join(figs_dir, 'generalization_scatter.png'))
    plot_radar_comparison(
        df_shift,
        save_path=os.path.join(figs_dir, 'radar_comparison.png'))
    plot_results_table(
        df_shift.drop(columns=[c for c in df_shift.columns
                                if c in ("inference_time_sec",
                                         "inference_samples_per_sec",
                                         "num_predictions",
                                         "training_time_sec")],
                      errors="ignore"),
        title="Domain Shift Summary Table",
        save_path=os.path.join(figs_dir, 'domain_shift_summary_table.png'))

    if available_timing_cols and len(available_timing_cols) > 1:
        plot_timing_comparison(
            df_timing,
            save_path=os.path.join(figs_dir, 'timing_comparison.png'))

    # ── Phase 6: Robustness ──────────────────────────────────────────────────
    logger.info("--- Phase 6: Robustness Testing ---")
    tester = PerturbationTesting(seed=config['seed'])
    robustness_results = []
    shuffle_results = []
    data_regime_results = []
    error_results = {}

    for name, model in models.items():
        # Error analysis (out-of-domain)
        out_preds = preds_out_all[name]
        _, _, errors_df = analyze_errors(
            y_out_true, out_preds, test_out_df['text'].tolist(),
            model_name=name, save_dir=reports_dir)
        error_results[name] = errors_df

        # Noise injection robustness
        for noise in config['robustness']['noise_levels']:
            logger.info(f"Robustness noise test: model='{name}' noise={noise}")
            perturbed_df = tester.create_perturbed_dataset(
                test_in_df, perturbation_type="noise", param=noise)
            preds = model.predict(perturbed_df['text'].tolist())
            m = compute_metrics(perturbed_df['label'].tolist(), preds)
            m['noise_level'] = noise
            m['model'] = name
            robustness_results.append(m)

        # Word-shuffle robustness (baseline vs shuffled)
        shuffled_df = tester.create_perturbed_dataset(
            test_in_df, perturbation_type="shuffle")
        baseline_preds = preds_in_all[name]
        shuffled_preds = model.predict(shuffled_df['text'].tolist())
        shuffle_results.append({
            "model": name,
            "baseline_accuracy": compute_metrics(
                y_in_true, baseline_preds)['accuracy'],
            "shuffled_accuracy": compute_metrics(
                y_in_true, shuffled_preds)['accuracy'],
        })

        # Data-regime evaluation
        for regime in config['robustness']['data_regimes']:
            scaled_df = tester.scale_data_regime(test_in_df, regime)
            logger.info(f"Data-regime eval: model='{name}' fraction={regime}")
            preds = model.predict(scaled_df['text'].tolist())
            m = compute_metrics(scaled_df['label'].tolist(), preds)
            m['data_regime'] = regime
            m['model'] = name
            data_regime_results.append(m)

    # Save CSVs
    df_rob = pd.DataFrame(robustness_results)
    df_rob.to_csv(os.path.join(results_dir, 'robustness_summary.csv'),
                  index=False)

    df_shuf = pd.DataFrame(shuffle_results)
    df_shuf.to_csv(os.path.join(results_dir, 'shuffle_robustness_summary.csv'),
                   index=False)

    if data_regime_results:
        df_regime = pd.DataFrame(data_regime_results)
        df_regime.to_csv(os.path.join(results_dir, 'data_regime_summary.csv'),
                         index=False)

    # ── Robustness & data-regime plots ───────────────────────────────────────
    logger.info("--- Generating robustness and data-regime plots ---")

    # NEW: single combined robustness figure (noise accuracy + noise F1 + shuffle)
    plot_robustness_combined(
        df_rob, df_shuf,
        save_path=os.path.join(figs_dir, 'robustness_combined.png'))

    plot_results_table(
        df_rob, title="Robustness Under Noise – All Models",
        save_path=os.path.join(figs_dir, 'robustness_table.png'))

    if data_regime_results:
        # NEW: single combined data-regime figure (accuracy + F1 side-by-side)
        plot_data_regime_combined(
            df_regime,
            save_path=os.path.join(figs_dir, 'data_regime_combined.png'))
        plot_results_table(
            df_regime, title="Data Regime Performance – All Models",
            save_path=os.path.join(figs_dir, 'data_regime_table.png'))

    # ── Error taxonomy ───────────────────────────────────────────────────────
    logger.info("--- Generating error taxonomy plots ---")
    plot_error_taxonomy_crossmodel(
        error_results,
        save_path=os.path.join(figs_dir, 'error_taxonomy_crossmodel.png'))

    logger.info("=" * 60)
    logger.info("Pipeline complete!  All outputs saved to:")
    logger.info(f"  Figures  → {figs_dir}/")
    logger.info(f"  CSVs     → {results_dir}/")
    logger.info(f"  Reports  → {reports_dir}/")


if __name__ == "__main__":
    main()