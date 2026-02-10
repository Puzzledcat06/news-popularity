from src.data.proxy_signals import urgency_score, emotion_intensity, readability_score, length_score

def explain_text(text):
    return {
        "urgency": urgency_score(text),
        "emotion": emotion_intensity(text),
        "readability": readability_score(text),
        "length_norm": length_score(text)
    }
