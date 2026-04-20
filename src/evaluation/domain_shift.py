import pandas as pd
from src.evaluation.metrics import compute_metrics


def evaluate_domain_shift(model, test_in_df, test_out_df, model_name="Model"):
    """Phase 7: Domain Shift Evaluation.

    Returns a tuple ``(results_dict, preds_in, preds_out)`` so callers can
    generate confusion matrices and per-domain detailed metrics.
    """
    preds_in = model.predict(test_in_df['text'].tolist())
    metrics_in = compute_metrics(test_in_df['label'].tolist(), preds_in)

    preds_out = model.predict(test_out_df['text'].tolist())
    metrics_out = compute_metrics(test_out_df['label'].tolist(), preds_out)

    acc_drop = metrics_in['accuracy'] - metrics_out['accuracy']
    f1_drop = metrics_in['f1_macro'] - metrics_out['f1_macro']

    results = {
        'model': model_name,
        'in_domain_accuracy': metrics_in['accuracy'],
        'out_domain_accuracy': metrics_out['accuracy'],
        'accuracy_drop': acc_drop,
        'in_domain_f1': metrics_in['f1_macro'],
        'out_domain_f1': metrics_out['f1_macro'],
        'f1_drop': f1_drop,
        'in_domain_precision': metrics_in['precision_macro'],
        'in_domain_recall': metrics_in['recall_macro'],
        'out_domain_precision': metrics_out['precision_macro'],
        'out_domain_recall': metrics_out['recall_macro'],
    }

    return results, preds_in, preds_out
