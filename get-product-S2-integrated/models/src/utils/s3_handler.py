from loguru import logger
import io 
import numpy as np
from PIL import Image
from urllib.parse import urlparse

from ..auth.auth import S3Connector
from ..utils.stac_client import get_product_content



def connect_to_s3(endpoint_url: str, access_key_id: str, secret_access_key: str) -> tuple:
    """Connect to S3 storage."""
    try:
        connector = S3Connector(
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region_name='default'
        )
        logger.success(f"Successfully connected to {endpoint_url} ")
        return connector.get_s3_resource(), connector.get_s3_client()
    except Exception as e:
        logger.error(f"Failed to connect to S3 storage: {e}")
        return None, None
    


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


def load_bands_from_s3(s3_client, bucket_name: str, item, bands: list, resize_shape: tuple = (1830, 1830), product_level: str ="L1C") -> np.ndarray:
    """Load bands from S3 storage."""
    try:
        band_data = []
        for band_name in bands:

            if product_level=="L1C":
                logger.info("Loading L1C bands from S3 storage")
                product_url = extract_s3_path_from_url(item.assets[band_name].href)
            else:
                band_name  = f"{band_name}_10m"
                logger.info("Loading L2A bands from S3 storage")
                product_url = extract_s3_path_from_url(item.assets[band_name].href)

            content = get_product_content(s3_client, bucket_name, product_url)
            image = Image.open(io.BytesIO(content)).resize(resize_shape)
            band_data.append(np.array(image))
        logger.success("Loaded bands from S3 storage")
        return np.dstack(band_data)
    except Exception as e:
        logger.error(f"Failed to load bands from S3 storage: {e}")
        return None