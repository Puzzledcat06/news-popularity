import argparse
from src.pipelines.training_pipeline import run_training_pipeline
from src.pipelines.inference_pipeline import run_inference_demo

def main(mode):
    if mode == "train":
        run_training_pipeline()
    elif mode == "infer":
        run_inference_demo()
    else:
        raise ValueError("Mode must be train or infer")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="infer")
    args = parser.parse_args()
    main(args.mode)
