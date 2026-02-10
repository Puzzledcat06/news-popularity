📰 News Popularity Intelligence System (Transformer-Based)
📌 Overview

Digital news platforms must decide which articles to highlight at publish time, even before real engagement metrics (clicks, shares, impressions) are available. This project implements a Transformer-based News Popularity Intelligence System that predicts the relative popularity potential of news articles using only their textual content (title + description), under cold-start conditions.

The system leverages pretrained Transformer models (DistilBERT) for deep semantic representation and combines them with proxy popularity signals (urgency, sentiment intensity, readability, length) as weak supervision to rank and score articles. The output includes explainable insights to support editorial decision-making.

🎯 Key Features

🔹 Transformer-based text representation (DistilBERT)

🔹 Popularity scoring & ranking without explicit labels (weak supervision)

🔹 Proxy popularity signals:

Urgency keywords

Emotional intensity (sentiment)

Readability

Text length normalization

🔹 Explainability: shows why an article is ranked higher

🔹 End-to-end pipeline (data → embeddings → scoring → inference)

🔹 Interactive Streamlit UI (local demo)

🧠 Problem Framing (Cold-Start Popularity)

Popularity is treated as a latent variable (not directly observable at publish time).
The system infers popularity potential based on linguistic and emotional cues such as:

Emotional intensity

Urgency and novelty

Linguistic clarity

Narrative style

This mirrors real-world newsroom workflows where content must be prioritized before user feedback exists.

🏗️ System Architecture (High Level)
Raw News Text (Title + Description)
        │
        ▼
Transformer Encoder (DistilBERT)
        │
        ▼
Dense Article Embeddings
        │
        ▼
Proxy Popularity Signals (Weak Supervision)
        │
        ▼
Lightweight Popularity Head (Scoring / Ranking)
        │
        ▼
Explainable Rankings + Scores (Streamlit UI)

📁 Project Structure
news-popularity-intelligence/
│
├── app/
│   └── streamlit_run.py        # Streamlit UI (3 pages)
│
├── src/
│   ├── models/                # Transformer embedder + popularity head
│   ├── pipelines/             # Training & inference pipelines
│   ├── data/                  # Proxy signal logic
│   ├── utils/                 # Logging, helpers
│   └── visualization/         # Explainability helpers
│
├── notebooks/                 # EDA, experiments, representation learning
├── config/                    # Configs (paths, params)
├── artifacts/                 # Model artifacts (ignored in Git)
├── main.py                    # Single entry point (train / infer)
├── requirements.txt
└── README.md

🚀 How to Run (Local)
1️⃣ Create Virtual Environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train (Optional)
python main.py --mode train

4️⃣ Run Inference Demo (CLI)
python main.py --mode infer

5️⃣ Launch Streamlit App (Local)
streamlit run app/streamlit_run.py

📊 Evaluation (Without Labels)

Since true popularity labels are unavailable, evaluation is performed using:

Qualitative ranking inspection

Case studies (breaking news vs routine news)

Ranking consistency across similar articles

Human-interpretable explanations

This aligns with real-world cold-start evaluation in recommender and media systems.

🧪 Tech Stack

Python

PyTorch

Hugging Face Transformers (DistilBERT)

NLTK, TextStat (proxy signals)

Streamlit (UI)

Pandas, NumPy

💼 Business Use Cases

Editorial content prioritization

Homepage & feed ranking

Breaking-news detection

Content amplification decisions

AI-assisted newsroom workflows

Media analytics support

🧩 Key Learnings

Handling unlabeled data using weak supervision

Applying Transformer embeddings for semantic representation

Designing ranking systems without explicit ground truth

Building explainable AI systems for editorial decision support

Structuring an ML project in a production-like pipeline

📌 Notes

Popularity scores reflect proxy-driven potential, not actual user engagement.

The system is intended for decision support under cold-start conditions.

Model artifacts and datasets are excluded from the repository for reproducibility and size constraints.


Write a 2–3 line project summary for your resume

Prepare interview explanation points for this project
