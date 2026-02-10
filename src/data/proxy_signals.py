from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat

sia = SentimentIntensityAnalyzer()
urgency_keywords = ["breaking", "urgent", "exclusive", "alert", "now"]

def urgency_score(text):
    return sum(1 for k in urgency_keywords if k in text.lower())

def sentiment_intensity(text):
    return emotion_intensity(text)

def emotion_intensity(text):
    return abs(sia.polarity_scores(text)["compound"])

def readability_score(text):
    return textstat.flesch_reading_ease(text)

def length_score(text):
    return min(len(text.split()) / 200.0, 2.0)

def compute_proxy_popularity(text):
    return (
        0.3 * urgency_score(text)
        + 0.3 * emotion_intensity(text)
        + 0.2 * length_score(text)
        + 0.2 * (1 / (1 + abs(readability_score(text))))
    )
