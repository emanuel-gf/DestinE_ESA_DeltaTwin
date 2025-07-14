import io
import os
import sys
import time
import warnings
import json
from functools import wraps
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from loguru import logger
import argparse

from src.utils import load_config
from src.plot import generate_plot_band, generate_tci_plot, extract_metrics_to_json

warnings.filterwarnings('ignore')

def parse_arguments() -> argparse.Namespace:
    """
    """
    parser = argparse.ArgumentParser(
        description="S3 bucket - Get Sentinel 2 Products - Select Bands and Product Level",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add positional arguments - correct syntax
    parser.add_argument('prep_np', type=str, help="Predicted Image. Numpy ndarray")
    parser.add_argument('x_np', type=str, help="Image used to predict - Sourced image. Numpy ndarray")
    parser.add_argument('gt_np', type=str, help="Ground Truth ndarray")
    parser.add_argument('bands', type=str, help="List of bands to be extracted. Should match the exact name of the catalogue. (e.g: 'B02,B03,B04'). Numpy ndarray")
    return parser.parse_args()


def parse_bands(bands_string: str) -> list[str]:
    """
    Parse a comma-separated string of bands into a list of strings.
    
    Args:
        bands_string: Comma-separated string of bands (e.g., "B02,B03,B04")
    
    Returns:
        List of band strings with whitespace stripped
    
    Example:
        >>> parse_bands("B02,B03,B04")
        ['B02', 'B03', 'B04']
        >>> parse_bands("B02, B03 , B04")
        ['B02', 'B03', 'B04']
    """
    return [band.strip() for band in bands_string.split(',')]


def initialize_env(prep_np:str, x_np: str, gt_np: str, bands=str) -> dict:
    """Load environment variables."""
    try:
        load_dotenv()
        logger.success("Loaded environment variables")
        return {
            "prep_np": str(prep_np),
            "x_np": str(x_np),
            "gt_np":str(gt_np),
            "bands":parse_bands(bands)
            }
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        return {}



def main() -> None:
    # Set up logging
    logger.add("generate_plot.log", rotation="10 MB")
    logger.info("Start workflow ...")

    # Load environment and configs
    args = parse_arguments()

    # Initialize environment with parsed arguments
    env = initialize_env(
        pred_np=args.pred_np,
        x_np=args.x_np, 
        gt_np= args.gt_np,
        bands_str = args.bands
    )

    dir_path = os.getcwd()

    # Fetch data
    bands = env["bands"]
    path_gt_np = env["gt_np"] 
    path_x_np = env["x_np"]
    path_pred_np = env["pred_np"]

    with np.load(path_gt_np) as a:
        gt_np = a["array"]
    
    with np.load(path_x_np) as a:
        x_np = a["array"]
    
    with np.load(path_pred_np) as a:
        pred_np = a["array"]


    ## Load data
    # Visualization
    generate_plot_band(x_np=x_np, gt_np=gt_np, pred_np=pred_np, bands=bands, cmap="Grays_r", output_dir='.')
    logger.info(f"Plot bands completed")
    generate_tci_plot(x_np=x_np, gt_np=gt_np, pred_np=pred_np, bands=bands[::-1], output_dir=".")
    logger.info(f"Plot TCI completed")
    extract_metrics_to_json(x_np=x_np, pred_np=pred_np, gt_np=gt_np, bands=bands, output_dir='.')
    logger.info(f"Metrics completed")



    logger.success("Workflow completed")



if __name__ == "__main__":
    main()
