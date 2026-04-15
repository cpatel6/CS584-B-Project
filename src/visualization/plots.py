import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# ──────────────────────────────────────────────────────────────────
# Helper – shared style
# ──────────────────────────────────────────────────────────────────
_DPI = 150
_PALETTE = "tab10"


def _save(fig, path):
    if path:
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────
# 1. Feature importance bar chart (for any linear model)
# ──────────────────────────────────────────────────────────────────
def plot_feature_importance(feature_names, coefs, top_n=20, save_path=None,
                            title="Top Features"):
    """Bar chart showing top positive and negative feature coefficients."""
    df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    df = df.reindex(df["coef"].abs().sort_values(ascending=False).index)
    plot_df = pd.concat([
        df[df["coef"] > 0].head(top_n),
        df[df["coef"] < 0].head(top_n),
    ])
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.35)))
    colors = ["#e74c3c" if c > 0 else "#3498db" for c in plot_df["coef"]]
    ax.barh(plot_df["feature"], plot_df["coef"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Coefficient")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────
# 2. Robustness / data-regime degradation line plot
# ──────────────────────────────────────────────────────────────────
def plot_degradation(results_df, metric="f1_macro", save_path=None):
    """Line plot for noise or data-regime degradation."""
    if results_df is None or results_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    if "noise_level" in results_df.columns:
        x_col, xlabel = "noise_level", "Noise Level"
        title = f"Model Robustness – {metric} vs. Noise Level"
    elif "data_regime" in results_df.columns:
        x_col, xlabel = "data_regime", "Fraction of Data"
        title = f"Model Data Efficiency – {metric} vs. Training Data Fraction"
    else:
        return
    sns.lineplot(data=results_df, x=x_col, y=metric, hue="model",
                 marker="o", ax=ax, palette=_PALETTE)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────
# 3. Confusion matrix heatmap (single model / domain)
# ──────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name="Model", domain="",
                          save_path=None):
    """Annotated confusion matrix for binary sentiment classification."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Negative", "Positive"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    title = f"{model_name} – Confusion Matrix"
    if domain:
        title += f"\n({domain})"
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────
# 4. Grouped bar – in-domain vs out-of-domain for one metric
# ──────────────────────────────────────────────────────────────────
def plot_metrics_bar(df_shift, metric="accuracy", save_path=None):
    """Grouped bar chart: in-domain vs out-of-domain for a given metric."""
    in_col, out_col = f"in_domain_{metric}", f"out_domain_{metric}"
    if in_col not in df_shift.columns or out_col not in df_shift.columns:
        return
    models = df_shift["model"].tolist()
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, df_shift[in_col], width,
                   label="In-Domain (Amazon)", color="#2980b9")
    bars2 = ax.bar(x + width / 2, df_shift[out_col], width,
                   label="Out-of-Domain (IMDb)", color="#e74c3c")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison: In-Domain vs Out-of-Domain "
                 f"{metric.replace('_', ' ').title()}")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()
    for bar in list(bars1) + list(bars2):
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────
# 5. Domain-shift drop heatmap
# ──────────────────────────────────────────────────────────────────
def plot_domain_shift_drop_heatmap(df_shift, save_path=None):
    """Heatmap of accuracy_drop and f1_drop across all models."""
    drop_cols = [c for c in ["accuracy_drop", "f1_drop"] if c in df_shift.columns]
    if not drop_cols:
        return
    plot_data = df_shift.set_index("model")[drop_cols]
    fig, ax = plt.subplots(figsize=(7, max(4, len(df_shift) * 0.7)))
    sns.heatmap(plot_data, annot=True, fmt=".3f", cmap="RdYlGn_r",
                linewidths=0.5, vmin=0, ax=ax)
    ax.set_title("Domain Shift Drops\n(Larger = Worse Generalization)")
    ax.set_ylabel("Model")
    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────
# 6. Full performance metrics heatmap (all models × all metrics)
# ──────────────────────────────────────────────────────────────────
def plot_all_metrics_heatmap(df_shift, save_path=None):
    """Full metrics heatmap across all models – great for slide/report tables."""
    metric_cols = ["in_domain_accuracy", "out_domain_accuracy", "accuracy_drop",
                   "in_domain_f1", "out_domain_f1", "f1_drop"]
    available = [c for c in metric_cols if c in df_shift.columns]
    if not available:
        return
    plot_data = df_shift.set_index("model")[available]
    fig, ax = plt.subplots(figsize=(14, max(4, len(df_shift) * 0.8)))
    sns.heatmap(plot_data, annot=True, fmt=".3f", cmap="coolwarm",
                linewidths=0.5, ax=ax)
    ax.set_title("Complete Model Performance Matrix")
    ax.set_xticklabels([c.replace("_", "\n") for c in available],
                       rotation=0, fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────
# 7. Timing comparison (training time + inference throughput)
# ──────────────────────────────────────────────────────────────────
def plot_timing_comparison(df_timing, save_path=None):
    """Side-by-side bar charts for training time and inference throughput."""
    if df_timing is None or df_timing.empty:
        return
    has_train = "training_time_sec" in df_timing.columns
    has_infer = "inference_samples_per_sec" in df_timing.columns
    ncols = int(has_train) + int(has_infer)
    if ncols == 0:
        return
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]
    col_idx = 0
    if has_train:
        axes[col_idx].barh(df_timing["model"], df_timing["training_time_sec"],
                           color="#3498db")
        axes[col_idx].set_xlabel("Seconds")
        axes[col_idx].set_title("Training Time")
        for i, v in enumerate(df_timing["training_time_sec"]):
            axes[col_idx].text(v * 1.01, i, f"{v:.1f}s",
                               va="center", fontsize=8)
        col_idx += 1
    if has_infer:
        axes[col_idx].barh(df_timing["model"],
                           df_timing["inference_samples_per_sec"],
                           color="#e67e22")
        axes[col_idx].set_xlabel("Samples / Second")
        axes[col_idx].set_title("Inference Throughput")
    fig.suptitle("Computational Efficiency Comparison", fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────
# 8. Robustness – accuracy (additional to F1)
# ──────────────────────────────────────────────────────────────────
def plot_robustness_accuracy(df_robustness, save_path=None):
    """Line plot of accuracy (not F1) across noise levels."""
    if df_robustness is None or df_robustness.empty:
        return
    if "accuracy" not in df_robustness.columns or "noise_level" not in df_robustness.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df_robustness, x="noise_level", y="accuracy",
                 hue="model", marker="o", ax=ax, palette=_PALETTE)
    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy vs. Noise Level (Robustness)")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────
# 9. Data-regime accuracy (additional to F1)
# ──────────────────────────────────────────────────────────────────
def plot_data_regime_accuracy(df_regime, save_path=None):
    """Line plot of accuracy across data fractions."""
    if df_regime is None or df_regime.empty:
        return
    if "accuracy" not in df_regime.columns or "data_regime" not in df_regime.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df_regime, x="data_regime", y="accuracy",
                 hue="model", marker="o", ax=ax, palette=_PALETTE)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Fraction of Data")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy vs. Training Data Fraction (Data Efficiency)")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────
# 10. Error taxonomy cross-model heatmap
# ──────────────────────────────────────────────────────────────────
def plot_error_taxonomy_crossmodel(error_results_dict, save_path=None):
    """Heatmap comparing error categories across all models."""
    categories = ["negation", "sarcasm", "domain_specific_words", "typos", "other"]
    rows = []
    for model_name, errors_df in error_results_dict.items():
        if errors_df is None or errors_df.empty:
            continue
        counts = errors_df["category"].value_counts().to_dict()
        row = {cat: counts.get(cat, 0) for cat in categories}
        row["model"] = model_name
        rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows).set_index("model")[categories]
    fig, ax = plt.subplots(figsize=(12, max(4, len(rows) * 0.8)))
    sns.heatmap(df, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, ax=ax)
    ax.set_title("Error Taxonomy Cross-Model Comparison")
    ax.set_ylabel("Model")
    ax.set_xlabel("Error Category")
    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────
# 11. Label distribution across dataset splits
# ──────────────────────────────────────────────────────────────────
def plot_label_distribution(datasets, save_path=None):
    """Grouped bar showing label balance per dataset split.

    datasets: dict of {split_name: DataFrame} – each DataFrame must have 'label'.
    """
    fig, axes = plt.subplots(1, len(datasets),
                              figsize=(4 * len(datasets), 5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    colors = ["#e74c3c", "#2ecc71"]
    for ax, (name, df) in zip(axes, datasets.items()):
        counts = df["label"].value_counts().sort_index()
        bars = ax.bar(["Negative", "Positive"][:len(counts)],
                      counts.values, color=colors[:len(counts)])
        ax.set_title(name)
        ax.set_ylabel("Count")
        for bar in bars:
            ax.annotate(f"{int(bar.get_height())}",
                        xy=(bar.get_x() + bar.get_width() / 2,
                            bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=10)
    fig.suptitle("Label Distribution Across Dataset Splits", fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────
# 12. In-domain vs out-of-domain scatter (generalization)
# ──────────────────────────────────────────────────────────────────
def plot_generalization_scatter(df_shift, save_path=None):
    """Scatter: in-domain vs out-of-domain accuracy & F1.

    Points above the y=x diagonal generalise better than the diagonal.
    """
    pairs = [("accuracy", "in_domain_accuracy", "out_domain_accuracy"),
             ("f1_macro", "in_domain_f1", "out_domain_f1")]
    available_pairs = [(label, ic, oc) for label, ic, oc in pairs
                       if ic in df_shift.columns and oc in df_shift.columns]
    if not available_pairs:
        return
    fig, axes = plt.subplots(1, len(available_pairs),
                              figsize=(7 * len(available_pairs), 6))
    if len(available_pairs) == 1:
        axes = [axes]
    for ax, (label, in_col, out_col) in zip(axes, available_pairs):
        all_vals = pd.concat([df_shift[in_col], df_shift[out_col]])
        lo = max(0.3, all_vals.min() - 0.05)
        hi = min(1.0, all_vals.max() + 0.05)
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4,
                label="Perfect generalisation (y=x)")
        colors = plt.cm.get_cmap(_PALETTE)(
            np.linspace(0, 1, len(df_shift)))
        for (_, row), c in zip(df_shift.iterrows(), colors):
            ax.scatter(row[in_col], row[out_col], s=100, color=c, zorder=5)
            ax.annotate(row["model"], (row[in_col], row[out_col]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"In-Domain {label.replace('_', ' ').title()}")
        ax.set_ylabel(f"Out-of-Domain {label.replace('_', ' ').title()}")
        ax.set_title(f"Generalisation – {label.replace('_', ' ').title()}")
        ax.legend(fontsize=8)
    fig.suptitle("In-Domain vs Out-of-Domain Generalisation Scatter",
                 fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────
# 13. Radar / spider chart
# ──────────────────────────────────────────────────────────────────
def plot_radar_comparison(df_shift, save_path=None):
    """Spider / radar chart comparing models across multiple performance axes."""
    metrics = ["in_domain_accuracy", "out_domain_accuracy",
               "in_domain_f1", "out_domain_f1"]
    available = [m for m in metrics if m in df_shift.columns]
    if not available:
        return
    labels = [m.replace("_", "\n") for m in available]
    N = len(available)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    cmap = plt.cm.get_cmap(_PALETTE)
    colors = cmap(np.linspace(0, 1, len(df_shift)))

    for (_, row), c in zip(df_shift.iterrows(), colors):
        values = [row[m] for m in available] + [row[available[0]]]
        ax.plot(angles, values, "o-", linewidth=2,
                label=row["model"], color=c)
        ax.fill(angles, values, alpha=0.1, color=c)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Radar Chart", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────
# 14. Robustness summary – faceted accuracy + F1 side by side
# ──────────────────────────────────────────────────────────────────
def plot_robustness_summary(df_robustness, save_path=None):
    """Dual-panel line plots for accuracy and F1 vs. noise level."""
    if df_robustness is None or df_robustness.empty:
        return
    if "noise_level" not in df_robustness.columns:
        return
    has_acc = "accuracy" in df_robustness.columns
    has_f1 = "f1_macro" in df_robustness.columns
    metrics = ([("accuracy", "Accuracy")] if has_acc else []) + \
              ([("f1_macro", "Macro F1")] if has_f1 else [])
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics),
                              figsize=(8 * len(metrics), 6), sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric, ylabel) in zip(axes, metrics):
        sns.lineplot(data=df_robustness, x="noise_level", y=metric,
                     hue="model", marker="o", ax=ax, palette=_PALETTE)
        ax.set_xlabel("Noise Level")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. Noise Level")
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=8)
    fig.suptitle("Robustness Under Noise Injection", fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────
# 15. Word-shuffle robustness degradation
# ──────────────────────────────────────────────────────────────────
def plot_shuffle_robustness(df_shuffle, save_path=None):
    """Bar chart: accuracy before vs after word-order shuffling per model."""
    if df_shuffle is None or df_shuffle.empty:
        return
    required = {"model", "baseline_accuracy", "shuffled_accuracy"}
    if not required.issubset(df_shuffle.columns):
        return
    x = np.arange(len(df_shuffle))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, df_shuffle["baseline_accuracy"], width,
           label="Baseline (clean)", color="#2980b9")
    ax.bar(x + width / 2, df_shuffle["shuffled_accuracy"], width,
           label="After word shuffle", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(df_shuffle["model"], rotation=25, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness to Word-Order Shuffling")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────
# 16. Precision-Recall-F1 per-model bar chart
# ──────────────────────────────────────────────────────────────────
def plot_precision_recall_f1(df_metrics, domain="In-Domain", save_path=None):
    """Grouped bar chart showing precision, recall, F1 per model."""
    needed = ["model", "precision_macro", "recall_macro", "f1_macro"]
    missing = [c for c in needed if c not in df_metrics.columns]
    if missing:
        return
    models = df_metrics["model"].tolist()
    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, df_metrics["precision_macro"], width,
           label="Precision", color="#3498db")
    ax.bar(x,          df_metrics["recall_macro"],    width,
           label="Recall",    color="#2ecc71")
    ax.bar(x + width,  df_metrics["f1_macro"],        width,
           label="F1 Macro",  color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"Precision, Recall and F1 Comparison ({domain})")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────
# 17. Summary report table as an image (for presentations)
# ──────────────────────────────────────────────────────────────────
def plot_results_table(df, title="Model Comparison Summary", save_path=None):
    """Render a DataFrame as a nicely-formatted image table."""
    numeric_cols = df.select_dtypes(include="number").columns
    rounded = df.copy()
    rounded[numeric_cols] = rounded[numeric_cols].round(4)

    fig_width = max(10, len(rounded.columns) * 1.6)
    fig_height = max(3, len(rounded) * 0.55 + 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=rounded.values,
        colLabels=rounded.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    # Style header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ecf0f1")
    ax.set_title(title, pad=15, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
