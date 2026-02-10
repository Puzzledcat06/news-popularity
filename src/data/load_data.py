import pandas as pd
import yaml

def load_news_data():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    path = cfg["data"]["raw_path"]
    df = pd.read_csv(path)
    df = df.rename(columns={"Title": "title", "Description": "description"})
    return df[["title", "description"]].dropna()
