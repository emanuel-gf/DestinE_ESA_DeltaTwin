import io
import os
import sys
import time
import warnings
from functools import wraps

import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger


warnings.filterwarnings('ignore')


def initialize_env(pred_tensor=sys.argv[1], origin_tensor=sys.argv[2], valid_mask=sys.argv[3]) -> dict:
    """Load environment variables."""
    try:
        load_dotenv()
        logger.success("Loaded environment variables")
        return {
            "pred_tensor": str(pred_tensor),
            "origin_tensor": str(origin_tensor),
            "valid_mask":str(valid_mask)
        }
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        return {}


def postprocess(x_tensor: torch.Tensor, pred_tensor: np.ndarray, valid_mask: np.ndarray) -> tuple:
    """Postprocess the prediction."""
    try:
        x_np = x_tensor.cpu().numpy()[0].transpose(1, 2, 0)
        x_np[~valid_mask] = 0.0
        pred_np = pred_tensor
        pred_np[~valid_mask] = 0.0
        # Make sure all values are clipped to 0-1
        x_np = np.clip(x_np, 0, 1)
        pred_np = np.clip(pred_np, 0, 1)
        logger.success("Postprocess model output successull")
        return x_np, pred_np
    except Exception as e:
        logger.error(f"Failed to postprocess model output: {e}")
        return None, None


def main() -> None:
    # Set up logging
    logger.add("log_AiSen2Cor.log", rotation="10 MB")
    logger.info("Start workflow ...")
    # Load environment and configs

    env = initialize_env(key_id=sys.argv[1], secret_key=sys.argv[2])
    dir_path = os.getcwd()


    # Load tensors
    ## Load origin tensor
    path_origin_tensor = env["origin_tensor"]
    origin_tensor = torch.load(path_origin_tensor)

    ## Load pred tensor
    path_pred_tensor = env["pred_tensor"]
    pred_tensor = torch.load(path_pred_tensor)

    ## Load mask
    path_valid_mask = env["valid_mask"]
    with np.load(path_valid_mask) as a:
        valid_mask = a["array"]


    x_np, pred_np = postprocess(x_tensor=origin_tensor, pred_tensor=pred_tensor, valid_mask=valid_mask)

    ## Save the postprocess
    path_x_np_post = "./x_post_np.npz"
    path_pred_np_post = "./pred_post_np.npz"
    np.save(path_x_np_post, x_np)
    np.save(path_pred_np_post, pred_np)

    logger.info(f"Stored at: {path_pred_np_post} & {path_pred_np_post}")

    logger.success("Workflow completed")



if __name__ == "__main__":
    main()
