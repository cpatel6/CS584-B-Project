import pandas as pd
from src.evaluation.metrics import compute_metrics

def evaluate_domain_shift(model, test_in_df, test_out_df, model_name="Model"):
    """
    Phase 7: Domain Shift Evaluation (Critical Section)
    Tracks Absolute Accuracy drop and Absolute Macro-F1 drop.
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
        'f1_drop': f1_drop
    }
    
    return results
