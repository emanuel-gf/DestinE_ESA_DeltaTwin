import io
from datetime import datetime, timedelta
from loguru import logger
import random
import requests
from pystac_client import Client
import os

def get_product_content(s3_client, bucket_name: str, object_url: str) -> bytes:
    """
    Download the content of a product from an S3 bucket.

    Args:
        s3_client: A boto3 S3 client object.
        bucket_name (str): The name of the S3 bucket.
        object_url (str): The path to the object within the bucket.

    Returns:
        bytes: The content of the downloaded file.

    Raises:
        Exception: If an error occurs during the download process.
    """
    logger.info(f"Downloading {object_url}")

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_url)
        content = response['Body'].read()
        logger.success(f"Successfully downloaded {object_url}")
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise

    return content
