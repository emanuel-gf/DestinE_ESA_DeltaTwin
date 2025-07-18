import os
import sys
import warnings
from loguru import logger
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pystac_client
import ast
import argparse
from typing import Dict, List, Union

from src.utils import load_config, create_stac_json
from src.data_query import data_query_most_recent 



warnings.filterwarnings('ignore')
def parse_bbox(bbox_str: str) -> List[float]:
    """
    Parse bbox string into list of coordinates.
    
    Args:
        bbox_str: Comma-separated string of coordinates "min_lon,min_lat,max_lon,max_lat"
        
    Returns:
        List of four float coordinates
        
    Raises:
        ValueError: If bbox format is invalid
    """
    try:
        coords = [float(x.strip()) for x in bbox_str.split(',')]
        if len(coords) != 4:
            raise ValueError(
                f"bbox must contain exactly 4 coordinates, got {len(coords)}"
            )
        
        min_lon, min_lat, max_lon, max_lat = coords
        
        # Validate coordinate ranges
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError(
                "Longitude must be between -180 and 180"
            )
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError(
                "Latitude must be between -90 and 90"
            )
        if min_lon >= max_lon:
            raise ValueError(
                "min_lon must be less than max_lon"
            )
        if min_lat >= max_lat:
            raise ValueError(
                "min_lat must be less than max_lat"
            )
            
        return coords
    except ValueError as e:
        raise ValueError(
            f"Invalid bbox format. Expected 'min_lon,min_lat,max_lon,max_lat': {e}"
        )
    

def parse_arguments() -> Dict[str, Union[List[float], int]]:
    """
    Parse command line arguments using sys.argv.
    
    Returns:
        Dictionary with parsed bbox and cloud cover
    """
    logger.info(f"Command line arguments: {sys.argv}")
    logger.info(f"Number of arguments: {len(sys.argv)}")
    
    # Default values
    default_bbox = "3.2833,45.3833,11.2,50.1833"
    default_max_cloud_cover = 20
    
    # Get bbox from sys.argv[1] or use default
    if len(sys.argv) > 1:
        bbox_str = sys.argv[1]
        logger.info(f"Using bbox from command line: {bbox_str}")
    else:
        bbox_str = default_bbox
        logger.info(f"Using default bbox: {bbox_str}")
    
    # Get max_cloud_cover from sys.argv[2] or use default
    if len(sys.argv) > 2:
        max_cloud_cover = int(sys.argv[2])
        logger.info(f"Using max_cloud_cover from command line: {max_cloud_cover}")
    else:
        max_cloud_cover = default_max_cloud_cover
        logger.info(f"Using default max_cloud_cover: {max_cloud_cover}")
    
    # Parse and validate bbox
    bbox = parse_bbox(bbox_str)
    
    return {
        "bbox": bbox,
        "max_cloud_cover": max_cloud_cover
    }

def initialize_env(bbox_coords: List[float], max_cloud_cover: int) -> Dict[str, Union[List[float], int]]:
    """
    Initialize environment variables
    
    Args:
        bbox_str: List of bounding box coordinates: min_lon,min_lat,max_lon,max_lat
        max_cloud_cover: Maximum cloud cover percentage
        
    Returns:
        Dictionary with parsed bbox and cloud cover
    """
    try:
        load_dotenv()
        logger.success("Loaded environment variables")

        
        return {
            "bbox": bbox_coords,
            "max_cloud_cover": max_cloud_cover
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        raise

def main() -> None:
    # Set up logging
    logger.info(f"Command line arguments: {sys.argv}")
    logger.add("STAC-Query.log", rotation="10 MB")
    logger.info("Start STAC QUERY workflow ...")

    # # Parse command line arguments
    args = parse_arguments()

    # Load environment and configs
    env = initialize_env(args["bbox"], args["max_cloud_cover"])

    dir_path = os.getcwd() 

    bbox = env["bbox"]
    max_cloud_cover = int(env["max_cloud_cover"])
    logger.info(f"Using bbox: {bbox}, max_cloud_cover: {max_cloud_cover}")
    
    #print("Printing checkpoint: ",bbox, max_cloud_cover)
    #start_date = env["start_date"] if "start_date" in env else query_cfg["query"]["start_date"]
    #end_date = env["end_date"] if "end_date" in env else query_cfg["query"]["end_date"]
    
    query_cfg = load_config(f"{dir_path}/src/cfg/query_config.yaml")

    # Setup
    stac_url = 'https://stac.dataspace.copernicus.eu/v1/'
    catalog = pystac_client.Client.open(stac_url)
    
    
    l1c_item, l2a_item  = data_query_most_recent(catalog, bbox,  max_cloud_cover)

    # Save data to JSON
    create_stac_json(l1c_item, l2a_item, output_dir=dir_path)

    # Log completion
    logger.info("STAC Query completed successfully.")

    logger.debug(f"Working directory: {os.getcwd()}")
    logger.debug(f"Files in directory: {os.listdir('.')}")



if __name__ == "__main__":
    main()

