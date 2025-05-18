import torch.nn as nn
from transformers import BertModel
import torch

class BERT_LSTM_Model(nn.Module):
    def __init__(self, hidden_dim=256, num_labels=3, lstm_layers=1, dropout=0.3):
        super(BERT_LSTM_Model, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_input = bert_output.last_hidden_state
        lstm_out, _ = self.lstm(lstm_input)
        out = lstm_out[:, 0, :]
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits
