import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(feature_names, coefs, top_n=20, save_path=None):
    """
    Mandatory visualization for classical ML model feature interpretability.
    """
    df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
    df = df.sort_values(by='coef', ascending=False)
    
    top_pos = df.head(top_n)
    top_neg = df.tail(top_n)
    
    plot_df = pd.concat([top_pos, top_neg])
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='coef', y='feature', data=plot_df, hue='feature', legend=False)
    plt.title('Top 20 Positive and Negative Features (TF-IDF/LR)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_degradation(results_df, save_path=None):
    """
    Degradation plot highlighting robustness falls.
    """
    plt.figure(figsize=(8, 6))
    if 'noise_level' in results_df.columns:
        sns.lineplot(data=results_df, x='noise_level', y='f1_macro', hue='model', marker='o')
        plt.title('Model Robustness to Noise Degradation')
    elif 'data_regime' in results_df.columns:
        sns.lineplot(data=results_df, x='data_regime', y='f1_macro', hue='model', marker='o')
        plt.title('Model Data Efficiency')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
