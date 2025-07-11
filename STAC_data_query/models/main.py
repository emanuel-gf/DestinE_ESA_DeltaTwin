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

def parse_arguments_2() -> Dict[str, Union[List[float], int]]:
    """
    Parse command line arguments using sys.argv.
    Expected arguments: min_lon min_lat max_lon max_lat [cloud_cover]
    """
    if len(sys.argv) < 5:
        raise ValueError("Usage: python script.py <min_lon> <min_lat> <max_lon> <max_lat> [cloud_cover]")
    
    try:
        # Parse the four bbox coordinates
        min_lon = float(sys.argv[1])
        min_lat = float(sys.argv[2])
        max_lon = float(sys.argv[3])
        max_lat = float(sys.argv[4])
        
        # Parse optional cloud cover (default to 20 if not provided)
        cloud_cover = int(sys.argv[5]) if len(sys.argv) > 5 else 20
        
        # Validate coordinate ranges
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if min_lon >= max_lon:
            raise ValueError("min_lon must be less than max_lon")
        if min_lat >= max_lat:
            raise ValueError("min_lat must be less than max_lat")
        if not (0 <= cloud_cover <= 100):
            raise ValueError("Cloud cover must be between 0 and 100")
        
        return {
            "bbox": [min_lon, min_lat, max_lon, max_lat],
            "cloud_cover": cloud_cover
        }
        
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid arguments: {e}")

def validate_cloud_cover(value: Union[str,int]) -> int:
    """
    Validate cloud cover percentage.
    
    Args:
        value: String representation of cloud cover percentage
        
    Returns:
        Integer cloud cover value
        
    Raises:
        argparse.ArgumentTypeError: If cloud cover is not between 0-100
    """
    try:
        cloud_cover = int(value)
        if not (0 <= cloud_cover <= 100):
            raise argparse.ArgumentTypeError(
                "Cloud cover must be between 0 and 100"
            )
        return cloud_cover
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Cloud cover must be an integer"
        )

def parse_bbox(bbox_str: str) -> List[float]:
    """
    Parse bbox string into list of coordinates.
    
    Args:
        bbox_str: Comma-separated string of coordinates "min_lon,min_lat,max_lon,max_lat"
        
    Returns:
        List of four float coordinates
        
    Raises:
        argparse.ArgumentTypeError: If bbox format is invalid
    """
    try:
        coords = [float(x.strip()) for x in bbox_str.split(',')]
        if len(coords) != 4:
            raise argparse.ArgumentTypeError(
                f"bbox must contain exactly 4 coordinates, got {len(coords)}"
            )
        
        min_lon, min_lat, max_lon, max_lat = coords
        
        # Validate coordinate ranges
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise argparse.ArgumentTypeError(
                "Longitude must be between -180 and 180"
            )
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise argparse.ArgumentTypeError(
                "Latitude must be between -90 and 90"
            )
        if min_lon >= max_lon:
            raise argparse.ArgumentTypeError(
                "min_lon must be less than max_lon"
            )
        if min_lat >= max_lat:
            raise argparse.ArgumentTypeError(
                "min_lat must be less than max_lat"
            )
            
        return coords
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid bbox format. Expected 'min_lon,min_lat,max_lon,max_lat': {e}"
        )
    

def parse_arguments() -> Dict[str, Union[List[float], int]]:
    """
    Parse command line arguments using argparse.
    
    Returns:
        Dictionary with parsed bbox and cloud cover
    """
    parser = argparse.ArgumentParser(
        description="Query at the STAC Catalog with a bounding box and cloud cover parameters. The Date is defined for the most recent image scene.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python main.py "3.2833,45.3833,11.2,50.1833" 20
                """
            )
            
    parser.add_argument(
        'bbox',
        type=parse_bbox,
        help='Bounding box coordinates as "min_lon,min_lat,max_lon,max_lat"'
    )
    
    parser.add_argument(
        'cloud_cover',
        type=validate_cloud_cover,
        nargs='?',
        default=20,
        help='Maximum cloud cover percentage (0-100). Default: 20'
    )

    args = parser.parse_args()
    return {
            "bbox": args.bbox,
            "cloud_cover": args.cloud_cover
        }

def initialize_env(bbox_coords: List[float], cloud_cover: float) -> Dict[str, Union[List[float], float]]:
    """
    Initialize environment variables
    
    Args:
        bbox_str: List of bounding box coordinates: min_lon,min_lat,max_lon,max_lat
        cloud_cover: Maximum cloud cover percentage
        
    Returns:
        Dictionary with parsed bbox and cloud cover
    """
    try:
        load_dotenv()
        logger.success("Loaded environment variables")

        
        return {
            "bbox": bbox_coords,
            "cloud_cover": cloud_cover
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        raise


def main() -> None:
    # Set up logging
    logger.add("STAC-Query.log", rotation="10 MB")
    logger.info("Start workflow ...")

    # Parse command line arguments
    args = parse_arguments()

    # Load environment and configs
    env = initialize_env(args["bbox"], args["cloud_cover"])

    dir_path = os.getcwd() 
    #print(dir_path)

    bbox = env["bbox"]
    max_cloud_cover = env["cloud_cover"]
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

