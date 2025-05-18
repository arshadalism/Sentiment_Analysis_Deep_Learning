                                                                                Sentiment Analysis Using Deep Learning


This project implements a hybrid BERT + BiLSTM model for sentiment analysis on textual data. The goal is to classify text into Positive, Negative, and Neutral sentiment categories.
Features:
  BERT embeddings for contextual word representation
  BiLSTM for sequential learning
  Multi-class sentiment classification (Positive, Negative, Neutral)
  Trained on a balanced dataset of 15,000 text samples (5,000 per class)
  Deployed using FastAPI for real-time inference

Model Architecture:
  BERT (Pretrained): For encoding sentences with contextual embeddings
  BiLSTM Layer: For learning temporal dependencies
  Dense + Softmax: For final sentiment classification

Dataset:
  Source: GitHub-hosted dataset
  Format: CSV

Structure:
  text: Input sentence or phrase
  label: Sentiment class (0 = Negative, 1 = Neutral, 2 = Positive)

Technologies Used:
  Python
  PyTorch
  Hugging Face Transformers
  FastAPI
  scikit-learn
  Pandas
  NumPy
