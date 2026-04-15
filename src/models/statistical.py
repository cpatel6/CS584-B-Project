from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

class TFIDFLR:
    """ Phase 2: TF-IDF + Logistic Regression """
    def __init__(self, max_features=10000, random_state=42):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)
    
    def fit(self, X, y):
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        
    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)
        
    def get_feature_importance(self):
        """ Used for interpretability mappings mandated in the implementation plan """
        feature_names = self.vectorizer.get_feature_names_out()
        coefs = self.model.coef_[0]
        return feature_names, coefs

class NGramSVM:
    """ Phase 2: n-grams + SVM """
    def __init__(self, max_features=10000, ngram_range=(1, 2), random_state=42):
        self.vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.model = LinearSVC(random_state=random_state, dual=False, max_iter=2000)
        
    def fit(self, X, y):
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        
    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)

    def get_feature_importance(self):
        """Feature importance from LinearSVC coefficients."""
        feature_names = self.vectorizer.get_feature_names_out()
        coefs = self.model.coef_[0]
        return feature_names, coefs

class LDAModel:
    """ Phase 2: Topic Modeling (LDA) """
    def __init__(self, n_components=2, max_features=5000, random_state=42):
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        self.model = LatentDirichletAllocation(n_components=n_components, random_state=random_state)
        self.topic_sentiment_map = None # Needs to be derived manually post-hoc
        
    def fit(self, X, y=None):
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec)
        
    def predict(self, X):
        # Simply returning topic distribution instead of classification for now
        X_vec = self.vectorizer.transform(X)
        return self.model.transform(X_vec)
