import yaml
import torch
from src.data.load_data import load_news_data
from src.data.preprocess import build_full_text
from src.data.proxy_signals import compute_proxy_popularity
from src.models.transformer import TransformerEmbedder
from src.models.popularity_head import PopularityHead
from src.utils.logger import setup_logger

logger = setup_logger()

def run_training_pipeline():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    df = load_news_data()
    texts = df.apply(lambda r: build_full_text(r["title"], r["description"]), axis=1).tolist()
    texts = texts[: cfg["training"]["sample_limit"]]

    embedder = TransformerEmbedder()
    X = embedder.encode(texts)

    y = torch.tensor([compute_proxy_popularity(t) for t in texts]).float().unsqueeze(1)

    model = PopularityHead()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    loss_fn = torch.nn.MSELoss()

    for epoch in range(cfg["training"]["epochs"]):
        opt.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        opt.step()
        logger.info(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), cfg["artifacts"]["model_path"])
    logger.info("Model saved.")
