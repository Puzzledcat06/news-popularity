import torch
from transformers import AutoTokenizer, AutoModel
import yaml

class TransformerEmbedder:
    def __init__(self):
        with open("config/config.yaml") as f:
            cfg = yaml.safe_load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["transformer_name"])
        self.model = AutoModel.from_pretrained(cfg["model"]["transformer_name"])
        self.model.eval()

    def encode(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]
