import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat

# Ensure vader lexicon is available (Streamlit Cloud safe)
def get_sentiment_analyzer():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon")
        return SentimentIntensityAnalyzer()

sia = get_sentiment_analyzer()

urgency_keywords = ["breaking", "urgent", "exclusive", "alert", "now"]

def urgency_score(text: str) -> int:
    text = text.lower()
    return sum(1 for k in urgency_keywords if k in text)

def emotion_intensity(text: str) -> float:
    return abs(sia.polarity_scores(text)["compound"])

def sentiment_intensity(text: str) -> float:
    # alias for clarity
    return emotion_intensity(text)

def readability_score(text: str) -> float:
    return textstat.flesch_reading_ease(text)

def length_score(text: str) -> float:
    return min(len(text.split()) / 200.0, 2.0)

def compute_proxy_popularity(text: str) -> float:
    urgency = urgency_score(text)
    emotion = emotion_intensity(text)
    length = length_score(text)
    readability = readability_score(text)

    # Lower readability score (very complex text) should reduce popularity
    readability_norm = 1 / (1 + abs(readability))

    score = (
        0.3 * urgency
        + 0.3 * emotion
        + 0.2 * length
        + 0.2 * readability_norm
    )
    return float(score)
