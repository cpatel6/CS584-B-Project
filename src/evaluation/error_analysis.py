import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns


NEGATION_PATTERNS = [r"\bnot\b", r"\bnever\b", r"\bno\b", r"n't\b"]
SARCASM_PATTERNS = [r"\byeah right\b", r"\bas if\b", r"/s\b"]
DOMAIN_PATTERNS = [
    r"\bshipping\b",
    r"\bdelivery\b",
    r"\bprime\b",
    r"\bmovie\b",
    r"\bfilm\b",
    r"\bepisode\b",
]
TYPO_PATTERN = r"(.)\1{2,}"


def categorize_error_text(text: str) -> str:
    text = str(text).lower()
    if any(re.search(p, text) for p in NEGATION_PATTERNS):
        return "negation"
    if any(re.search(p, text) for p in SARCASM_PATTERNS):
        return "sarcasm"
    if any(re.search(p, text) for p in DOMAIN_PATTERNS):
        return "domain_specific_words"
    if re.search(TYPO_PATTERN, text):
        return "typos"
    return "other"

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
    errors = errors.copy()
    errors["category"] = errors["text"].apply(categorize_error_text)
    
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

        f.write("\nError Taxonomy Counts:\n")
        counts = errors["category"].value_counts().to_dict()
        for k, v in counts.items():
            f.write(f"- {k}: {v}\n")

    taxonomy_path = os.path.join(save_dir, f"{model_name}_error_taxonomy.csv")
    errors[["text", "true_label", "pred_label", "category"]].to_csv(
        taxonomy_path, index=False
    )

    counts_df = (
        errors["category"].value_counts().rename_axis("category").reset_index(name="count")
    )
    if not counts_df.empty:
        plt.figure(figsize=(8, 4))
        sns.heatmap(
            counts_df.set_index("category")[["count"]],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
        )
        plt.title(f"{model_name} Error Taxonomy Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{model_name}_error_taxonomy_heatmap.png"))
        plt.close()
            
    return fp, fn, errors
