import torch
import torch.nn as nn

class CNNText(nn.Module):
    """ Phase 4: CNN Text Classifier """
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes=(2, 3, 4), num_filters=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x).unsqueeze(1) # (batch, 1, seq_len, embed_dim)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs] # [(batch, num_filters, seq_len-k+1)]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # [(batch, num_filters)]
        x = torch.cat(x, 1) # (batch, len(filter_sizes)*num_filters)
        x = self.dropout(x)
        return self.fc(x)

class BiLSTMText(nn.Module):
    """ Phase 4: BiLSTM Text Classifier """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x, lengths=None):
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x)
        # Using hidden state of the last time step from both directions
        hn = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1) 
        hn = self.dropout(hn)
        return self.fc(hn)

class NeuralTextClassifier:
    """ Wrapper for CNN and BiLSTM to provide scikit-learn like fit/predict and GloVe integration """
    def __init__(self, model_class, glove_path=None, vocab_size=10000, embed_dim=100, max_seq_length=128, epochs=3, batch_size=32, device="cuda", **kwargs):
        from sklearn.feature_extraction.text import CountVectorizer
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.glove_path = glove_path
        
        self.vectorizer = CountVectorizer(max_features=vocab_size - 2) # leaves room for PAD (0) and UNK (vocab_size-1)
        self.model_class = model_class
        self.kwargs = kwargs
        self.model = None

    def _load_glove_matrix(self):
        import numpy as np
        import os
        embedding_matrix = np.random.uniform(-0.1, 0.1, (self.vocab_size, self.embed_dim))
        embedding_matrix[0] = 0.0 # PAD is zero
        
        if self.glove_path and os.path.exists(self.glove_path):
            vocab = self.vectorizer.vocabulary_
            glove_found = 0
            with open(self.glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    if word in vocab:
                        idx = vocab[word] + 1 # +1 because 0 is PAD
                        vector = np.asarray(values[1:], dtype="float32")
                        if len(vector) == self.embed_dim:
                            embedding_matrix[idx] = vector
                            glove_found += 1
            print(f"Loaded {glove_found} GloVe vectors into Neural Model.")
            
        return torch.tensor(embedding_matrix, dtype=torch.float32)

    def _tokenize_pad(self, X):
        X_tokenized = []
        analyzer = self.vectorizer.build_analyzer()
        vocab = self.vectorizer.vocabulary_
        for x in X:
            tokens = analyzer(x)
            indices = [vocab.get(t, self.vocab_size - 2) + 1 for t in tokens] 
            if len(indices) > self.max_seq_length:
                indices = indices[:self.max_seq_length]
            else:
                indices = indices + [0] * (self.max_seq_length - len(indices))
            X_tokenized.append(indices)
        return torch.tensor(X_tokenized, dtype=torch.long)

    def fit(self, X, y):
        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset
        
        self.vectorizer.fit(X)
        X_tensor = self._tokenize_pad(X)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = self.model_class(vocab_size=self.vocab_size, embed_dim=self.embed_dim, num_classes=2, **self.kwargs)
        
        # Integrate GloVe
        embedding_matrix = self._load_glove_matrix()
        self.model.embedding.weight.data.copy_(embedding_matrix)
        # Optional: Disable requires_grad if pure frozen GloVe is required. Usually we fine-tune embeddings.
        # self.model.embedding.weight.requires_grad = False
        
        self.model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
    def predict(self, X):
        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset
        self.model.eval()
        X_tensor = self._tokenize_pad(X)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch_x, in dataloader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        return np.array(preds)
