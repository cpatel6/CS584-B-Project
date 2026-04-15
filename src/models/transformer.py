from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class DistilBERTClassifier:
    """ Phase 5: Contextual Embeddings (Transformer) Fine-tuning """
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)
        
    def tokenize(self, texts, max_length=256):
        """ Model-specific tokenization as per implementation plan """
        return self.tokenizer(
            texts, 
            padding="max_length", 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        
    # Freezing/Unfreezing logic for fine-tuning ablation
    def freeze_base_model(self):
        for param in self.model.distilbert.parameters():
            param.requires_grad = False
            
    def unfreeze_base_model(self):
        for param in self.model.distilbert.parameters():
            param.requires_grad = True

    def fit(self, X, y, epochs=1, batch_size=16):
        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        # Batch tokenize to save memory if needed, but doing it in chunks
        all_input_ids = []
        all_attention_mask = []
        
        for i in range(0, len(X), batch_size):
            batch_texts = X[i:i+batch_size]
            tokens = self.tokenize(batch_texts)
            all_input_ids.append(tokens['input_ids'])
            all_attention_mask.append(tokens['attention_mask'])
            
        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_mask, dim=0)
        labels = torch.tensor(y, dtype=torch.long)
        
        dataset = TensorDataset(input_ids, attention_mask, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for batch in dataloader:
                b_input_ids, b_attention_mask, b_labels = [t.to(self.device) for t in batch]
                optimizer.zero_grad()
                outputs = self.model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

    def predict(self, X, batch_size=16):
        import numpy as np
        self.model.eval()
        preds = []
        
        for i in range(0, len(X), batch_size):
            batch_texts = X[i:i+batch_size]
            tokens = self.tokenize(batch_texts)
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(batch_preds)
                
        return np.array(preds)
