import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


import streamlit as st

from src.pipelines.inference_pipeline import (
    score_single_article,
    rank_articles,
    fetch_latest_news
)

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="News Popularity Intelligence",
    layout="wide"
)

# -----------------------
# Sidebar Navigation
# -----------------------
st.sidebar.title("Navigate")
page = st.sidebar.radio("Go to", ["Home", "News Intelligence", "Model Reasoning"])

# -----------------------
# HOME PAGE
# -----------------------
if page == "Home":
    st.title("📰 News Popularity Intelligence System")

    st.markdown("""
    **Problem**  
    At publishing time, real popularity signals (clicks, shares, impressions) are unavailable  
    (**cold-start problem**).

    **Solution**  
    This system estimates **popularity potential from content alone** using:
    - Transformer-based semantic representations (DistilBERT)
    - Weak supervision (urgency, emotion, readability, length)
    - Lightweight deep learning scoring head

    **Architecture**
    - DistilBERT encoder → CLS embeddings  
    - DL popularity head → popularity score  
    - Proxy signals → explainability

    **Use Cases**
    - Editorial content prioritization  
    - Homepage & feed ranking  
    - Breaking news identification  
    - AI-assisted newsroom workflows  
    """)

# -----------------------
# NEWS INTELLIGENCE PAGE
# -----------------------
elif page == "News Intelligence":
    st.title("🧠 News Intelligence")

    left, right = st.columns([1.2, 1])

    # ---------------- LEFT: Manual Scoring ----------------
    with left:
        st.subheader("Score a News Article")

        with st.container():
            title = st.text_input("News Title", placeholder="e.g. Breaking: Major Earthquake Hits City")
            description = st.text_area("News Description", height=150, placeholder="Short summary or description of the news article")

            if st.button("Predict Popularity", use_container_width=True):
                if not title or not description:
                    st.warning("Please enter both title and description.")
                else:
                    result = score_single_article(title, description)
                    st.metric("Predicted Popularity Score", round(result["score"], 3))
                    st.caption("Explanation (Proxy Signals)")
                    st.json(result["explanation"])

        st.markdown("### Try Sample Headlines")
        sample_cols = st.columns(3)
        samples = [
            ("Breaking: Major Earthquake Hits City", "A powerful earthquake struck the city early morning causing widespread damage."),
            ("Company Releases Quarterly Earnings", "The company reported its quarterly earnings with moderate growth."),
            ("Exclusive Interview With Film Star", "The famous actor shares insights about upcoming projects.")
        ]

        for col, (t, d) in zip(sample_cols, samples):
            if col.button(t, use_container_width=True):
                result = score_single_article(t, d)
                st.metric("Predicted Popularity Score", round(result["score"], 3))
                st.caption("Explanation (Proxy Signals)")
                st.json(result["explanation"])

    # ---------------- RIGHT: Live News ----------------
    with right:
        st.subheader("Live News")
        st.caption("Fetch real-time headlines and rank them by predicted popularity")

        query = st.text_input("Search Query", "breaking news")

        if st.button("Fetch & Rank Live News", use_container_width=True):
            articles = fetch_latest_news(query=query, page_size=5)

            if not articles:
                st.info("Live news unavailable. Add NEWS_API_KEY in Streamlit secrets to enable.")
            else:
                ranked = rank_articles(articles)

                for idx, r in enumerate(ranked, 1):
                    with st.container(border=True):
                        st.markdown(f"**{idx}. {r['title']}**")
                        st.write(f"Popularity Score: `{round(r['score'], 3)}`")
                        st.caption("Why this ranked this way:")
                        st.json(r["explanation"])

# -----------------------
# MODEL REASONING PAGE
# -----------------------
elif page == "Model Reasoning":
    st.title("🔍 Model Reasoning & Explainability")

    st.markdown("""
    ### How scores are produced
    1. Title and description are concatenated  
    2. Tokenized using DistilBERT tokenizer  
    3. Encoded using DistilBERT → CLS embedding  
    4. Lightweight DL head predicts popularity potential  

    ### Why some articles rank higher
    - Urgency-related language  
    - Emotional intensity  
    - Linguistic clarity (readability)  
    - Optimal content length  

    ### Evaluation without labels
    Since no real popularity labels are used:
    - Qualitative case studies  
    - Ranking consistency checks  
    - Comparative examples across different news styles  

    > ⚠️ **Note:**  
    These rankings reflect *proxy-driven popularity potential*, not actual user engagement.
    The system supports editorial prioritization under cold-start conditions.
    """)
