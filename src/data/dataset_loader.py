from datasets import load_dataset
import pandas as pd
from typing import Tuple

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
        dataset = load_dataset(self.config['data']['in_domain'])
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
        dataset = load_dataset(self.config['data']['out_of_domain'])
        test_ds = dataset['test']
        
        # Use ood_samples if available, fallback to test_samples or original len
        ood_count = self.config['data'].get('ood_samples', self.config['data'].get('test_samples', 1000))
        
        test_ds = test_ds.shuffle(seed=self.config['seed']).select(range(min(len(test_ds), ood_count)))
            
        test_df = test_ds.to_pandas()
        
        # IMDb text column is 'text'
        test_df = self._normalize_columns(test_df, source_text_col='text')
        
        return test_df
