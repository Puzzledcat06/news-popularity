import logging
import logging.config
import yaml

def setup_logger():
    with open("config/logging.yaml", "r") as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    return logging.getLogger("news_popularity")
