import numpy as np
from sklearn.linear_model import LogisticRegression

class GloVeLR:
    """
    Phase 3: Representation Learning (Static Embeddings)
    GloVe + Logistic Regression
    """
    def __init__(self, glove_path=None, embedding_dim=100, random_state=42):
        self.embedding_dim = embedding_dim
        self.embeddings_dict = {}
        
        # Load Glove if provided, otherwise assume caller sets them up or it's a dummy run
        if glove_path:
            self._load_glove(glove_path)
            
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        
    def _load_glove(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
                
    def _sentence_embedding(self, text):
        words = str(text).split()
        embs = [self.embeddings_dict[w] for w in words if w in self.embeddings_dict]
        if len(embs) == 0:
            return np.zeros(self.embedding_dim)
        # Pooling strategy (mean aggregation) as per Phase 3 plan
        return np.mean(embs, axis=0) 
        
    def fit(self, X, y):
        X_vec = np.array([self._sentence_embedding(t) for t in X])
        self.model.fit(X_vec, y)
        
    def predict(self, X):
        X_vec = np.array([self._sentence_embedding(t) for t in X])
        return self.model.predict(X_vec)
