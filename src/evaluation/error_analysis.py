import pandas as pd
import os

def analyze_errors(y_true, y_pred, texts, model_name, save_dir):
    """
    Phase 8: Qualitative Analysis - Document representative failure cases.
    """
    df = pd.DataFrame({
        'text': texts,
        'true_label': y_true,
        'pred_label': y_pred
    })
    
    # Filter where predictions are wrong
    errors = df[df['true_label'] != df['pred_label']]
    
    # False Positives (True = 0, Pred = 1)
    fp = errors[(errors['true_label'] == 0) & (errors['pred_label'] == 1)]
    # False Negatives (True = 1, Pred = 0)
    fn = errors[(errors['true_label'] == 1) & (errors['pred_label'] == 0)]
    
    fp_sample = fp.head(10)
    fn_sample = fn.head(10)
    
    report_path = os.path.join(save_dir, f"{model_name}_error_analysis.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Error Analysis for {model_name}\n")
        f.write("="*40 + "\n\n")
        
        f.write("False Positives (Predicted Positive, Actually Negative):\n")
        for idx, row in fp_sample.iterrows():
            f.write(f"- {row['text']}\n")
            
        f.write("\nFalse Negatives (Predicted Negative, Actually Positive):\n")
        for idx, row in fn_sample.iterrows():
            f.write(f"- {row['text']}\n")
            
    return fp, fn
