import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('Sentiment_dataset.csv')
df = df.groupby('sentiment').head(5000)  # Balance classes
sentiment_mapping = {"positive": 2, 'neutral': 1, 'negative': 0}
df['Sentiment_Value'] = df['sentiment'].map(sentiment_mapping)

# Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['selected_text'], df['Sentiment_Value'], test_size=0.2, random_state=42
)

# Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts.astype(str)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts.iloc[index])
        label = self.labels.iloc[index]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_length = 128
batch_size = 16
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# Hybrid Model: BERT + BiLSTM
class BERT_LSTM_Model(nn.Module):
    def __init__(self, hidden_dim=256, num_labels=3, lstm_layers=1, dropout=0.3):
        super(BERT_LSTM_Model, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # Freeze BERT (optional — unfreeze later if needed)
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        lstm_input = bert_output.last_hidden_state  # (batch_size, seq_len, 768)
        lstm_out, _ = self.lstm(lstm_input)  # (batch_size, seq_len, hidden_dim*2)
        out = lstm_out[:, 0, :]  # Use the first token's output
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits


# Initialize model
model = BERT_LSTM_Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 10
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    total_loss = 0
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

# Evaluation
model.eval()
true_labels = []
pred_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

# Metrics
accuracy = accuracy_score(true_labels, pred_labels)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=["Negative", "Neutral", "Positive"]))
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, pred_labels))
