import re
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class Preprocessor:
    """
    Mandatory Unified Preprocessing Rules as defined in Phase 1
    """
    def __init__(self, config: dict):
        self.config = config
        self.stop_words = set(stopwords.words('english'))
        self.max_seq_length = (
            self.config.get('preprocessing', {}).get('max_seq_length')
            or self.config.get('data', {}).get('max_seq_length')
        )
        
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
            
        # Lowercasing
        if self.config['preprocessing'].get('lowercase', True):
            text = text.lower()
            
        # Punctuation handling
        if self.config['preprocessing'].get('remove_punctuation', True):
            text = text.translate(str.maketrans('', '', string.punctuation))
            
        # Stopword handling
        if self.config['preprocessing'].get('remove_stopwords', True):
            tokens = text.split()
            tokens = [t for t in tokens if t not in self.stop_words]
            text = " ".join(tokens)
            
        # Clean extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Optional coarse token-length normalization (whitespace-based).
        # Final model tokenization still happens in model-specific pipelines.
        if self.max_seq_length:
            tokens = text.split()
            text = " ".join(tokens[: self.max_seq_length])
        
        return text

    def process_dataframe(self, df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
        df = df.copy()
        df[text_col] = df[text_col].apply(self.clean_text)
        return df
