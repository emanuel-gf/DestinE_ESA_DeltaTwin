import io
import os
import sys
import time
import warnings
from functools import wraps

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

from model_zoo.models import define_model
from utils.torch import load_model_weights
from utils.utils import load_config, load_numpy_as_tensor
warnings.filterwarnings('ignore')


def initialize_env(s2_preprocessed=sys.argv[1]) -> dict:
    """Load environment variables."""
    try:
        load_dotenv()
        logger.success("Loaded environment variables")
        return {
            "s2_preprocessed": str(s2_preprocessed)
        }
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        return {}


def load_model(model_cfg: dict, weights_path: str, device: torch.device) -> torch.nn.Module:
    """Load the model."""
    try:
        model = define_model(
            name=model_cfg["model_name"],
            encoder_name=model_cfg["encoder_name"],
            in_channel=model_cfg["in_channel"],
            out_channels=model_cfg["out_channels"],
            activation=model_cfg["activation"]
        )
        model = load_model_weights(model, filename=weights_path)
        logger.success("Model Loaded")
        return model.to(device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def predict(model: torch.nn.Module, x_tensor: torch.Tensor) -> np.ndarray:
    """Make a prediction."""
    try:
        model.eval()
        with torch.no_grad():
            pred = model(x_tensor)
        logger.success("L2A generation successfull")
        return pred.cpu().numpy()[0].transpose(1, 2, 0)
    except Exception as e:
        logger.error(f"Failed to generate L2A: {e}")
        return None

def main() -> None:
    # Set up logging
    logger.add("log_inference_prediction.log", rotation="10 MB")
    logger.info("Start Inference workflow ...")
    # Load environment and configs

    env = initialize_env(s2_preprocessed=sys.argv[1])
    dir_path = os.getcwd()

    model_cfg = load_config(f"{dir_path}/cfg/config.yaml")
    query_cfg = load_config(f"{dir_path}/cfg/query_config.yaml")
    model_path = f"{dir_path}/weight/AiSen2Cor_EfficientNet_b2.pth"

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_cfg["MODEL"], model_path, device)

    # Load the preprocessed x data as a tensor
    path_tensor = env["s2_preprocessed"]
    x_tensor = load_numpy_as_tensor(path_tensor, device=device)
    
    logger.info("Predicting")
    pred_tensor = predict(model=model, x_tensor=x_tensor) 

    ## Save the prediction 
    saved_path_tensor = "./pred_tensor.npz"
    
    np.savez_compressed(saved_path_tensor, pred_tensor)
    
    logger.debug(f"Tensor saved at {saved_path_tensor}")
    logger.success("Workflow completed")



if __name__ == "__main__":
    main()
