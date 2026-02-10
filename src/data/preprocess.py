import re

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_full_text(title, description):
    return clean_text(title) + " [SEP] " + clean_text(description)
