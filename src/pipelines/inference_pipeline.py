import torch
import numpy as np
import requests
import streamlit as st
from transformers import AutoTokenizer, AutoModel

from src.models.popularity_head import PopularityHead
from src.data.proxy_signals import (
    urgency_score,
    sentiment_intensity,
    readability_score,
    length_score
)

# -----------------------
# Load Transformer
# -----------------------
DEVICE = "cpu"
MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
transformer = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
transformer.eval()

# -----------------------
# Load Popularity Head
# -----------------------
def load_popularity_head(model_path="artifacts/models/popularity_head.pt"):
    head = PopularityHead(input_dim=768)
    state = torch.load(model_path, map_location=DEVICE)
    head.load_state_dict(state)
    head.eval()
    return head

popularity_head = load_popularity_head()

# -----------------------
# Text → Embedding
# -----------------------
def get_embedding(text: str):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = transformer(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# -----------------------
# Score Single Article
# -----------------------
def score_single_article(title: str, description: str):
    full_text = f"{title} [SEP] {description}"

    emb = get_embedding(full_text)

    with torch.no_grad():
        score = popularity_head(emb).item()

    explanation = {
        "urgency": urgency_score(full_text),
        "emotion": sentiment_intensity(full_text),
        "readability": readability_score(full_text),
        "length_norm": length_score(full_text),
    }

    return {
        "title": title,
        "score": float(score),
        "explanation": explanation
    }

# -----------------------
# Rank Multiple Articles
# -----------------------
def rank_articles(articles):
    results = []

    for a in articles:
        res = score_single_article(a["title"], a["description"])
        results.append(res)

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results

# -----------------------
# NewsAPI Integration (Optional)
# -----------------------
def fetch_latest_news(query="breaking news", page_size=5):
    api_key = st.secrets.get("NEWS_API_KEY", None)

    if api_key is None:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": page_size,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": api_key
    }

    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
    except Exception as e:
        print("NewsAPI error:", e)
        return []

    articles = []
    for a in data.get("articles", []):
        if a.get("title") and a.get("description"):
            articles.append({
                "title": a["title"],
                "description": a["description"]
            })

    return articles
