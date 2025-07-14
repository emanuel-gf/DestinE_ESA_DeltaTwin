import yaml
from loguru import logger
from urllib.parse import urlparse
import pandas as pd
import os


def load_config(config_path: str = "cfg/config.yaml") -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): The path to the configuration YAML file.

    Returns:
        dict: The loaded configuration dictionary.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.success(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML configuration: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}


def extract_s3_path_from_url(url: str) -> str:
    """
    Extract the S3 object path from an S3 URL or URI.

    This function parses S3 URLs/URIs and returns just the object path portion,
    removing the protocol (s3://), bucket name, and any leading slashes.

    Args:
        url (str): The full S3 URI (e.g., 's3://eodata/path/to/file.jp2')

    Returns:
        str: The S3 object path (without protocol, bucket name and leading slashes)

    Raises:
        ValueError: If the provided URL is not an S3 URL.
    """
    if not url.startswith('s3://'):
        return url

    parsed_url = urlparse(url)

    if parsed_url.scheme != 's3':
        raise ValueError(f"URL {url} is not an S3 URL")

    object_path = parsed_url.path.lstrip('/')
    return object_path


def remove_last_segment_rsplit(sentinel_id: str) -> str:
    """
    Remove the last segment from a Sentinel ID by splitting at the last underscore.

    Args:
        sentinel_id (str): The Sentinel ID to process.

    Returns:
        str: The Sentinel ID without the last segment.
    """
    parts = sentinel_id.rsplit('_', 1)
    return parts[0]
