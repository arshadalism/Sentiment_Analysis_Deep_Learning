import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from model_class import BERT_LSTM_Model

app = FastAPI(title="Sentiment Analysis")

# Load model and tokenizer
model = BERT_LSTM_Model()
model.load_state_dict(torch.load("bert_lstm_model.pth", map_location=torch.device("cpu")))
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str


# Sentiment mapping
sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}


@app.post("/predict/")
async def predict_sentiment(input_data: TextInput):
    import torch
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    """
    Predict the sentiment of the given text.
    """
    try:
        # Tokenize input text
        encoding = tokenizer(
            input_data.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        # Move tensors to the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        encoding = {key: val.to(device) for key, val in encoding.items()}

        # Predict
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_class = torch.argmax(logits, dim=1).cpu().item()

        # Map prediction to sentiment
        sentiment = sentiment_mapping[predicted_class]

        return {"text": input_data.text, "sentiment": sentiment}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#
if __name__ == '__main__':
     uvicorn.run("backend:app", reload=True)
