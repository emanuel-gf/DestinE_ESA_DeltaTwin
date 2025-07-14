import os
import sys
from functools import wraps

import cv2
import numpy as np
from dotenv import load_dotenv
from loguru import logger

from utils.utils import load_config,save_tensor_as_numpy

def initialize_env(path_ndarray_raw_s2=sys.argv[1]) -> dict:
    """Load environment variables."""
    try:
        load_dotenv()
        logger.success("Loaded environment variables")
        return {
            "s2_raw": str(path_ndarray_raw_s2),
        }
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        return {}


def normalize(data_array: np.ndarray) -> tuple:
    """Normalize the data array."""
    try:
        normalized_data, valid_masks = [], []
        for i in range(data_array.shape[2]):
            band = data_array[:, :, i]
            valid_mask = band > 0
            norm_band = band.astype(np.float32)
            norm_band[valid_mask] /= 10000
            norm_band = np.clip(norm_band, 0, 1)
            norm_band[~valid_mask] = 0
            normalized_data.append(norm_band)
            valid_masks.append(valid_mask)
        logger.success("Normalized data array")
        return np.dstack(normalized_data), np.dstack(valid_masks)
    except Exception as e:
        logger.error(f"Failed to normalize data array: {e}")
        return None, None


def preprocess(raw_data: np.ndarray, resize: int):
    """Preprocess the raw data."""
    try:
        x_data, valid_mask = normalize(raw_data)
        x_data = cv2.resize(x_data, (resize, resize), interpolation=cv2.INTER_AREA)
        valid_mask = cv2.resize(valid_mask.astype(np.uint8), (resize, resize), interpolation=cv2.INTER_NEAREST).astype(bool)
       #x_tensor = torch.from_numpy(x_data).float().permute(2, 0, 1).unsqueeze(0).to(device) # [B , C , W, H]
        logger.success("Preprocess raw data successull")
        return x_data, valid_mask
    except Exception as e:
        logger.error(f"Failed to preprocess raw data: {e}")
        return None, None


def main() -> None:
    # Set up logging
    logger.add("log_preprocessing.log", rotation="10 MB")
    logger.info("Start Preprocessing workflow ...")

    # Load environment and configs
    env = initialize_env(path_ndarray_raw_s2=sys.argv[1])

    dir_path = os.getcwd()
    model_cfg = load_config(f"{dir_path}/cfg/config.yaml")

    resize = model_cfg["TRAINING"]["resize"]

    ## Load the S2 raw
    s2_raw_path = env["s2_raw"]
    with np.load(s2_raw_path) as data:
        s2_raw = data['array']

    # Preprocess
    ## It is not returning a tensor, only a ndarray. The convertion to tensor is on Inferece Module. 
    ## This was done to eliminate pytorch from this docker.
    x_np_tensor, valid_mask = preprocess(raw_data=s2_raw, resize=resize, device=device)

    ## Save the x_np_tensor on numpy compressed format
    ## Saved the mask as a compressed numpy .npz
    saved_path_tensor = "./s2_preprocessed.npz"
    saved_path_ndarray = "./valid_mask_preprocessed.npz"

    np.savez_compressed(saved_path_tensor, x_np_tensor)

    np.savez_compressed(saved_path_ndarray,
                        valid_mask)
    
    logger.success("Workflow completed")



if __name__ == "__main__":
    main()
