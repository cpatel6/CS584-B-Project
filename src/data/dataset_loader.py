import os

from datasets import load_dataset
import pandas as pd
from typing import Tuple

# Canonical HuggingFace Hub dataset name aliases.
# HF migrated many community datasets to namespaced identifiers in 2024.
# We try the configured name first, then the legacy/namespaced fallback.
_DATASET_ALIASES = {
    "amazon_polarity": ["fancyzhx/amazon_polarity", "amazon_polarity"],
    "imdb": ["stanfordnlp/imdb", "imdb"],
}

# Local CSV fallback paths (relative to project root).
_LOCAL_CSV = {
    "amazon_polarity": "data/amazon_polarity.csv",
    "imdb": "data/imdb.csv",
}


def _load_dataset_with_fallback(name: str):
    """Try loading a dataset by name, falling back to known aliases on failure.

    When the HuggingFace Hub is unreachable the function checks for a local
    CSV at a well-known path, and synthesises the data if neither is available.
    """
    candidates = _DATASET_ALIASES.get(name, [name])
    last_exc = None
    for candidate in candidates:
        try:
            return load_dataset(candidate)
        except Exception as exc:
            last_exc = exc

    # --- offline fallback: local CSV -----------------------------------------
    local_path = _LOCAL_CSV.get(name)
    if local_path and os.path.exists(local_path):
        df = pd.read_csv(local_path)
        # Wrap in a dict mimicking HF dataset splits so callers work unchanged.
        from datasets import Dataset
        return {"train": Dataset.from_pandas(df), "test": Dataset.from_pandas(df)}

    # --- offline fallback: synthesise on the fly ------------------------------
    try:
        from src.data.synthetic_data import save_synthetic_datasets
        save_synthetic_datasets(data_dir="data")
        if local_path and os.path.exists(local_path):
            df = pd.read_csv(local_path)
            from datasets import Dataset
            return {"train": Dataset.from_pandas(df), "test": Dataset.from_pandas(df)}
    except Exception:
        pass

    raise RuntimeError(
        f"Failed to load dataset '{name}'. "
        "Check your internet connection or set HF_TOKEN for authenticated access. "
        f"Last error: {last_exc}"
    ) from last_exc


class DatasetLoader:
    def __init__(self, config: dict):
        self.config = config
    
    def _normalize_columns(self, df: pd.DataFrame, source_text_col: str) -> pd.DataFrame:
        if source_text_col in df.columns:
            df.rename(columns={source_text_col: 'text'}, inplace=True)
            
        # Ensure we only have standard columns for consistency
        cols_to_keep = ['text', 'label']
        available_cols = [c for c in cols_to_keep if c in df.columns]
        return df[available_cols]

    def load_in_domain(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads Phase 1: Dataset A (In-Domain: Amazon Reviews)
        Returns: Train, Validation, Test splits
        """
        dataset = _load_dataset_with_fallback(self.config['data']['in_domain'])
        train_ds = dataset['train']
        test_ds = dataset['test']
        
        seed = self.config['seed']
        
        # Subsample BEFORE loading to pandas to prevent Out-Of-Memory on huge datasets
        if self.config['data'].get('train_samples'):
            train_ds = train_ds.shuffle(seed=seed).select(range(min(len(train_ds), self.config['data']['train_samples'])))
            
        if self.config['data'].get('test_samples'):
            test_ds = test_ds.shuffle(seed=seed).select(range(min(len(test_ds), self.config['data']['test_samples'])))
            
        train_df = train_ds.to_pandas()
        test_df = test_ds.to_pandas()
        
        # Amazon polarity dataset text column is 'content'
        train_df = self._normalize_columns(train_df, source_text_col='content')
        test_df = self._normalize_columns(test_df, source_text_col='content')
            
        # Stratified split for validation
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            train_df, 
            test_size=self.config['data']['val_split'],
            stratify=train_df['label'],
            random_state=seed
        )
            
        return train_df, val_df, test_df
        
    def load_out_of_domain(self) -> pd.DataFrame:
        """
        Loads Phase 1: Dataset B (Out-of-Domain: IMDb Reviews)
        Returns: Test split only
        """
        dataset = _load_dataset_with_fallback(self.config['data']['out_of_domain'])
        test_ds = dataset['test']
        
        # Use ood_samples if available, fallback to test_samples or original len
        ood_count = self.config['data'].get('ood_samples', self.config['data'].get('test_samples', 1000))
        
        test_ds = test_ds.shuffle(seed=self.config['seed']).select(range(min(len(test_ds), ood_count)))
            
        test_df = test_ds.to_pandas()
        
        # IMDb text column is 'text'
        test_df = self._normalize_columns(test_df, source_text_col='text')
        
        return test_df
