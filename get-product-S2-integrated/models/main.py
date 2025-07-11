import io
import os
import sys
import time
import warnings
from typing import List,Dict,Union


import numpy as np
import pandas as pd
import pystac_client
import pystac
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
import json
import argparse

from src.auth.auth import S3Connector
from src.utils.utils import extract_s3_path_from_url, load_config


from src.utils.s3_handler import connect_to_s3, load_bands_from_s3


warnings.filterwarnings('ignore')

def parse_arguments() -> argparse.Namespace:
    """
    """
    parser = argparse.ArgumentParser(
        description="S3 bucket - Get Sentinel 2 Products - Select Bands and Product Level",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add positional arguments - correct syntax
    parser.add_argument('cdse_key', type=str, help="CDSE Key")
    parser.add_argument('cdse_secret', type=str, help="CDSE Secret Value")
    parser.add_argument('bands', type=str, help="List of bands to be extracted. Should match the exact name of the catalogue. (e.g: 'B02,B03,B04')")
    parser.add_argument('product_level', type=str, help="Product Level regarding S3 catalogue. (e.g: L1C OR L2A)")
    parser.add_argument('stac_item', type=str, help="Path to a json containing STAC Item dict objects.")
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


def initialize_env(key_id:str, secret_key: str, bands_str: str, product_level=str, stac_item=str) -> dict:
    """Load environment variables."""
    try:
        load_dotenv()
        logger.success("Loaded environment variables")
        return {
            "access_key_id": str(key_id),
            "secret_access_key": str(secret_key),
            "bands":parse_bands(bands_str),
            "product_level":str(product_level),
            "stac_item":str(stac_item)
        }
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        return {}



def main() -> None:
    # Set up logging
    logger.add("Get-SEN2-Product-from-S3bucket.log", rotation="10 MB")
    logger.info("Start workflow ...")

    # Load environment and configs
    args = parse_arguments()

    # Initialize environment with parsed arguments
    env = initialize_env(
        key_id=args.cdse_key,
        secret_key=args.cdse_secret, 
        bands_str = args.bands,
        product_level= args.product_level,
        stac_item = args.stac_item
    )

    dir_path = os.getcwd()
    #print(dir_path)

    # Fetch Data
    bands = env["bands"]
    product_level = env["product_level"]
    path_stac_json = env["stac_item"]

    logger.debug(f"Name of the variables from environment: Bands:{bands} Product_level{product_level} Stac_json{path_stac_json}")
    
    model_cfg = load_config(f"{dir_path}/src/cfg/config.yaml")
    query_cfg = load_config(f"{dir_path}/src/cfg/query_config.yaml")

    # Setup
    endpoint_url = query_cfg["endpoint_url"]
    bucket_name = query_cfg["bucket_name"]

    ## Connect S3 
    s3, s3_client = connect_to_s3(endpoint_url, env["access_key_id"], env["secret_access_key"])
 

    ## Load the catalog given by the DEPENDECIE - STAC-DATA-QUERY
    with open(path_stac_json, 'r') as file:
        data_stac = json.load(file)
    
    ## Capital Letters
    product_level = str(product_level.upper())
    Sentinel_item = pystac.Item.from_dict(data_stac[product_level])
    #l2a_item = pystac.Item.from_dict(data_stac['L2A'])

    ## Load from S3 - SERVICE
    Sentinel_raw_data = load_bands_from_s3(s3_client, bucket_name, Sentinel_item, bands, product_level=product_level)
    #l2a_raw_data = load_bands_from_s3(s3_client, bucket_name, l2a_item, bands, product_level="L2A")

    ## Save the S2 image as a np compressed format. At the Root 
    filename_output = f"./{product_level}_raw"
    
    logger.info(f'Saving the files at: {filename_output}')

    try:
        np.savez_compressed(filename_output, array=Sentinel_raw_data)
        #np.savez_compressed(filename_l2a, array=l2a_raw_data)
    
    except Exception as e:
        logger.error(f"Failed to generate json: {e}")

    logger.success("Workflow - Get Product - completed!")


if __name__ == "__main__":
    main()
