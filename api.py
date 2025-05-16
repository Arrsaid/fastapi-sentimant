# --- Configuration de l'environnement ---
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
warnings.filterwarnings('ignore')         

from transformers import logging
logging.set_verbosity_error()             

# --- Imports FastAPI & modèles ---
from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from utils import preprocess_tweet_bert

# --- Chargement du modèle et du tokenizer ---
model_path = "./bert_model"
model = TFBertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# --- Initialisation de l'application ---
app = FastAPI(title="API BERT - Analyse de sentiment")

# --- Schéma de requête ---
class Tweet(BaseModel):
    text: str

# --- Route de prédiction ---
@app.post("/predict")
def predict(tweet: Tweet):
    tweet = preprocess_tweet_bert(tweet.text)
    inputs = tokenizer(
        tweet,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=128
    )
    outputs = model(inputs)
    logits = outputs.logits.numpy()
    proba = tf.nn.softmax(logits, axis=1).numpy()[0]
    pred_class = np.argmax(proba)

    sentiment = "positif" if pred_class == 1 else "négatif"
    confidence = round(float(proba[pred_class]), 4)

    return {
        "sentiment": sentiment,
        "confidence": confidence
    }
