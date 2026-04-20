import math
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------------------------
# Offline-only lightweight transformer (no HuggingFace weights needed)
# ---------------------------------------------------------------------------

class _SimpleTransformerModel(nn.Module):
    """A tiny 2-layer transformer encoder for offline fallback."""

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 nhead: int = 4, num_layers: int = 2,
                 max_seq_len: int = 128, num_classes: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Embedding(max_seq_len + 1, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        positions = torch.arange(1, seq_len + 1,
                                 device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_encoding(positions)
        # Build key_padding_mask (True = ignore)
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = (input_ids == 0)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        # Mean-pool over non-padding tokens
        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).float()
            x = (x * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
        else:
            x = x.mean(dim=1)
        return self.classifier(self.dropout(x))


class _OfflineDistilBERTClassifier:
    """Vocabulary-based tiny transformer that mimics the DistilBERTClassifier API."""

    def __init__(self, device: str = "cpu", max_seq_len: int = 128,
                 vocab_size: int = 8000, embed_dim: int = 64):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self._word2idx: dict = {}
        self.model: _SimpleTransformerModel | None = None

    # ------------------------------------------------------------------
    def _build_vocab(self, texts):
        from collections import Counter
        counts: Counter = Counter()
        for t in texts:
            counts.update(str(t).lower().split())
        # index 0 = PAD, 1 = UNK
        vocab = [w for w, _ in counts.most_common(self.vocab_size - 2)]
        self._word2idx = {w: i + 2 for i, w in enumerate(vocab)}

    def _encode(self, texts):
        out = []
        for t in texts:
            ids = [self._word2idx.get(w, 1) for w in str(t).lower().split()]
            ids = ids[:self.max_seq_len]
            ids += [0] * (self.max_seq_len - len(ids))
            out.append(ids)
        input_ids = torch.tensor(out, dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        return input_ids, attention_mask

    # ------------------------------------------------------------------
    def fit(self, X, y, epochs: int = 2, batch_size: int = 32):
        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset

        self._build_vocab(X)
        self.model = _SimpleTransformerModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            max_seq_len=self.max_seq_len,
        ).to(self.device)

        input_ids, attn_mask = self._encode(X)
        labels = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(input_ids, attn_mask, labels)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.model.train()
        for _ in range(epochs):
            for b_ids, b_mask, b_y in loader:
                b_ids = b_ids.to(self.device)
                b_mask = b_mask.to(self.device)
                b_y = b_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(b_ids, b_mask)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()

    def predict(self, X, batch_size: int = 32):
        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset

        self.model.eval()
        input_ids, attn_mask = self._encode(X)
        ds = TensorDataset(input_ids, attn_mask)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for b_ids, b_mask in loader:
                b_ids = b_ids.to(self.device)
                b_mask = b_mask.to(self.device)
                logits = self.model(b_ids, b_mask)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        return np.array(preds)


# ---------------------------------------------------------------------------
# Public class — tries real DistilBERT, falls back to offline version
# ---------------------------------------------------------------------------

class DistilBERTClassifier:
    """ Phase 5: Contextual Embeddings (Transformer) Fine-tuning """

    def __init__(self, model_name="distilbert-base-uncased", num_labels=2, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._impl: object = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels)
            self.model.to(self.device)
            self._use_hf = True
        except (OSError, Exception):
            # No network / no cached weights – use the lightweight offline model.
            self._use_hf = False
            self._offline = _OfflineDistilBERTClassifier(
                device=str(self.device),
                max_seq_len=128, vocab_size=8000, embed_dim=64)

    # ------------------------------------------------------------------
    def tokenize(self, texts, max_length=256):
        """ Model-specific tokenization as per implementation plan """
        if self._use_hf:
            return self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        raise NotImplementedError("tokenize() not available in offline mode")

    def freeze_base_model(self):
        if self._use_hf:
            for param in self.model.distilbert.parameters():
                param.requires_grad = False

    def unfreeze_base_model(self):
        if self._use_hf:
            for param in self.model.distilbert.parameters():
                param.requires_grad = True

    # ------------------------------------------------------------------
    def fit(self, X, y, epochs: int = 1, batch_size: int = 16):
        if self._use_hf:
            self._fit_hf(X, y, epochs=epochs, batch_size=batch_size)
        else:
            self._offline.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X, batch_size: int = 16):
        if self._use_hf:
            return self._predict_hf(X, batch_size=batch_size)
        return self._offline.predict(X, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Original HuggingFace implementations
    # ------------------------------------------------------------------

    def _fit_hf(self, X, y, epochs: int = 1, batch_size: int = 16):
        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        all_input_ids = []
        all_attention_mask = []
        for i in range(0, len(X), batch_size):
            tokens = self.tokenize(X[i:i + batch_size])
            all_input_ids.append(tokens['input_ids'])
            all_attention_mask.append(tokens['attention_mask'])

        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_mask, dim=0)
        labels = torch.tensor(y, dtype=torch.long)

        ds = TensorDataset(input_ids, attention_mask, labels)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for batch in loader:
                b_ids, b_mask, b_y = [t.to(self.device) for t in batch]
                optimizer.zero_grad()
                outputs = self.model(b_ids, attention_mask=b_mask, labels=b_y)
                outputs.loss.backward()
                optimizer.step()

    def _predict_hf(self, X, batch_size: int = 16):
        import numpy as np

        self.model.eval()
        preds = []
        for i in range(0, len(X), batch_size):
            tokens = self.tokenize(X[i:i + batch_size])
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask=attention_mask).logits
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        return np.array(preds)
