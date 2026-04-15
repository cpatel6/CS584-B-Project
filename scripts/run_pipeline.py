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
from src.visualization.plots import plot_feature_importance, plot_degradation
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


def run_pipeline_all_models(config, train_df, test_in_df, test_out_df):
    all_shift_results = []
    trained_models = {}
    
    # === STATISTICAL MODELS ===
    
    # 1. TF-IDF + Logistic Regression
    logger.info("--- Training Statistical Model 1: TF-IDF + LR ---")
    tfidf_lr = TFIDFLR(max_features=config['preprocessing']['max_features_tfidf'], random_state=config['seed'])
    train_time = measure_training_time(tfidf_lr.fit, train_df['text'].tolist(), train_df['label'].tolist())
    trained_models['TFIDF_LR'] = tfidf_lr
    res1 = evaluate_domain_shift(tfidf_lr, test_in_df, test_out_df, model_name="TFIDF_LR")
    inference_metrics = measure_inference_speed(tfidf_lr.predict, test_in_df['text'].tolist())
    res1.update(inference_metrics)
    res1["training_time_sec"] = train_time
    all_shift_results.append(res1)
    _save_model_checkpoint(tfidf_lr, "TFIDF_LR", config['paths']['models_dir'])
    
    features, coefs = tfidf_lr.get_feature_importance()
    plot_feature_importance(features, coefs, save_path=os.path.join(config['paths']['figures_dir'], 'tfidf_lr_feature_importance.png'))
    
    # 2. N-Grams + SVM
    logger.info("--- Training Statistical Model 2: N-Grams + SVM ---")
    ngram_svm = NGramSVM(max_features=config['preprocessing']['max_features_tfidf'], ngram_range=(1, 2), random_state=config['seed'])
    train_time = measure_training_time(ngram_svm.fit, train_df['text'].tolist(), train_df['label'].tolist())
    trained_models['NGram_SVM'] = ngram_svm
    res2 = evaluate_domain_shift(ngram_svm, test_in_df, test_out_df, model_name="NGram_SVM")
    inference_metrics = measure_inference_speed(ngram_svm.predict, test_in_df['text'].tolist())
    res2.update(inference_metrics)
    res2["training_time_sec"] = train_time
    all_shift_results.append(res2)
    _save_model_checkpoint(ngram_svm, "NGram_SVM", config['paths']['models_dir'])
    
    # 3. GloVe + LR (Treated purely as fixed embedding source for statistical models)
    logger.info("--- Training Statistical Model 3: GloVe + LR ---")
    glove_lr = GloVeLR(glove_path=None, random_state=config['seed']) # Will use random initialized dummies if None
    train_time = measure_training_time(glove_lr.fit, train_df['text'].tolist(), train_df['label'].tolist())
    trained_models['GloVe_LR'] = glove_lr
    res3 = evaluate_domain_shift(glove_lr, test_in_df, test_out_df, model_name="GloVe_LR")
    inference_metrics = measure_inference_speed(glove_lr.predict, test_in_df['text'].tolist())
    res3.update(inference_metrics)
    res3["training_time_sec"] = train_time
    all_shift_results.append(res3)
    _save_model_checkpoint(glove_lr, "GloVe_LR", config['paths']['models_dir'])

    # === NEURAL MODELS ===
    
    # 4. CNNText + GloVe Embeddings
    logger.info("--- Training Neural Model 1: CNNText ---")
    cnn_model = NeuralTextClassifier(
        CNNText, 
        glove_path=None, 
        epochs=config['training']['epochs'], 
        device=config['training']['device']
    )
    train_time = measure_training_time(cnn_model.fit, train_df['text'].tolist(), train_df['label'].tolist())
    trained_models['CNNText_GloVe'] = cnn_model
    res4 = evaluate_domain_shift(cnn_model, test_in_df, test_out_df, model_name="CNNText_GloVe")
    inference_metrics = measure_inference_speed(cnn_model.predict, test_in_df['text'].tolist())
    res4.update(inference_metrics)
    res4["training_time_sec"] = train_time
    all_shift_results.append(res4)
    _save_model_checkpoint(cnn_model, "CNNText_GloVe", config['paths']['models_dir'])
    
    # 5. BiLSTM + GloVe Embeddings
    logger.info("--- Training Neural Model 2: BiLSTM ---")
    bilstm_model = NeuralTextClassifier(
        BiLSTMText, 
        glove_path=None, 
        hidden_size=64, 
        epochs=config['training']['epochs'], 
        device=config['training']['device']
    )
    train_time = measure_training_time(bilstm_model.fit, train_df['text'].tolist(), train_df['label'].tolist())
    trained_models['BiLSTM_GloVe'] = bilstm_model
    res5 = evaluate_domain_shift(bilstm_model, test_in_df, test_out_df, model_name="BiLSTM_GloVe")
    inference_metrics = measure_inference_speed(bilstm_model.predict, test_in_df['text'].tolist())
    res5.update(inference_metrics)
    res5["training_time_sec"] = train_time
    all_shift_results.append(res5)
    _save_model_checkpoint(bilstm_model, "BiLSTM_GloVe", config['paths']['models_dir'])

    # 6. DistilBERT Fine-tuned
    logger.info("--- Training Neural Model 3: DistilBERT ---")
    distilbert_model = DistilBERTClassifier(device=config['training']['device'])
    train_time = measure_training_time(
        distilbert_model.fit,
        train_df['text'].tolist(), 
        train_df['label'].tolist(), 
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
    )
    trained_models['DistilBERT'] = distilbert_model
    res6 = evaluate_domain_shift(distilbert_model, test_in_df, test_out_df, model_name="DistilBERT")
    inference_metrics = measure_inference_speed(distilbert_model.predict, test_in_df['text'].tolist())
    res6.update(inference_metrics)
    res6["training_time_sec"] = train_time
    all_shift_results.append(res6)
    _save_model_checkpoint(distilbert_model, "DistilBERT", config['paths']['models_dir'])
    
    return all_shift_results, trained_models


def main():
    config = load_config()
    ensure_dirs(config)
    seed_everything(config['seed'])
    
    logger.info("Phase 1: Dataset Pipeline & Preprocessing")
    loader = DatasetLoader(config)
    train_df, val_df, test_in_df = loader.load_in_domain()
    test_out_df = loader.load_out_of_domain()
    
    preprocessor = Preprocessor(config)
    train_df['text'] = train_df['text'].fillna('')
    test_in_df['text'] = test_in_df['text'].fillna('')
    test_out_df['text'] = test_out_df['text'].fillna('')
    
    train_df = preprocessor.process_dataframe(train_df)
    test_in_df = preprocessor.process_dataframe(test_in_df)
    test_out_df = preprocessor.process_dataframe(test_out_df)
    
    # Run pipelines for ALL Models
    domain_shift_results, models = run_pipeline_all_models(config, train_df, test_in_df, test_out_df)
    
    # Save Domain Shift summary table
    df_shift = pd.DataFrame(domain_shift_results)
    df_shift.to_csv(os.path.join(config['paths']['results_dir'], 'domain_shift_summary.csv'), index=False)
    logger.info(f"\nFinal Comparative Domain Shift Metrics:\n{df_shift.to_string()}")
    timing_cols = ["model", "training_time_sec", "inference_time_sec", "inference_samples_per_sec", "num_predictions"]
    available_timing_cols = [c for c in timing_cols if c in df_shift.columns]
    if available_timing_cols:
        df_shift[available_timing_cols].to_csv(
            os.path.join(config['paths']['results_dir'], 'timing_summary.csv'),
            index=False,
        )
    
    # Robustness testing across ALL models
    logger.info("--- Starting Phase 6: Robustness Testing across all models ---")
    tester = PerturbationTesting(seed=config['seed'])
    robustness_results = []
    data_regime_results = []
    
    for name, model in models.items():
        # Error analysis on cross-domain behavior
        out_preds = model.predict(test_out_df['text'].tolist())
        analyze_errors(
            test_out_df['label'].tolist(),
            out_preds,
            test_out_df['text'].tolist(),
            model_name=name,
            save_dir=config['paths']['reports_dir'],
        )

        for noise in config['robustness']['noise_levels']:
            logger.info(f"Testing model '{name}' on noise level: {noise}")
            perturbed_df = tester.create_perturbed_dataset(test_in_df, perturbation_type="noise", param=noise)
            preds = model.predict(perturbed_df['text'].tolist())
            metrics = compute_metrics(perturbed_df['label'].tolist(), preds)
            metrics['noise_level'] = noise
            metrics['model'] = name
            robustness_results.append(metrics)

        for regime in config['robustness']['data_regimes']:
            scaled_df = tester.scale_data_regime(test_in_df, regime)
            logger.info(f"Evaluating model '{name}' on data regime: {regime}")
            preds = model.predict(scaled_df['text'].tolist())
            metrics = compute_metrics(scaled_df['label'].tolist(), preds)
            metrics['data_regime'] = regime
            metrics['model'] = name
            data_regime_results.append(metrics)
             
    df_res = pd.DataFrame(robustness_results)
    df_res.to_csv(os.path.join(config['paths']['results_dir'], 'robustness_summary.csv'), index=False)
    plot_degradation(df_res, save_path=os.path.join(config['paths']['figures_dir'], 'robustness_degradation.png'))
    if data_regime_results:
        df_regime = pd.DataFrame(data_regime_results)
        df_regime.to_csv(os.path.join(config['paths']['results_dir'], 'data_regime_summary.csv'), index=False)
        plot_degradation(df_regime, save_path=os.path.join(config['paths']['figures_dir'], 'data_regime_scaling.png'))
    
    logger.info("Pipeline execution completed for all models!")
    logger.info("Check outputs/results/domain_shift_summary.csv and outputs/figures/robustness_degradation.png for visual comparisons.")

if __name__ == "__main__":
    main()
