import datetime
import pandas as pd
import numpy as np
from loguru import logger
import os

#from ..utils.benchmark import benchmark

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

def test_import():
    print("Import successful!")
    print("benchmark module:", benchmark)

    

def data_query(catalog, bbox: list, start_date: str, end_date: str, max_cloud_cover: int):
    """
    Fetch both L1C and L2A products from CDSE STAC catalog and find matching pairs.

    Args:
        catalog: STAC catalog client
        bbox: Bounding box coordinates [west, south, east, north]
        start_date: Start date in format "YYYY-MM-DD"
        end_date: End date in format "YYYY-MM-DD"
        max_cloud_cover: Maximum cloud cover percentage

    Returns:
        tuple: (matched L1C item, matched L2A item)
    """
    try:
        # Search for L1C products
        logger.info(f"Searching for L1C products from {start_date} to {end_date} in bbox {bbox}")
        l1c_items = catalog.search(
            collections=['sentinel-2-l1c'],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            max_items=1000
        ).item_collection()

        # Search for L2A products
        logger.info(f"Searching for L2A products from {start_date} to {end_date} in bbox {bbox}")
        l2a_items = catalog.search(
            collections=['sentinel-2-l2a'],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=1000,
        ).item_collection()

        # Filter L2A items to remove those with high nodata percentage
        l2a_items = [item for item in l2a_items if item.properties.get("statistics", {}).get('nodata', 100) < 5]

        # Convert to dataframes for easier matching
        l1c_dicts = [item.to_dict() for item in l1c_items]
        l2a_dicts = [item.to_dict() for item in l2a_items]

        df_l1c = pd.DataFrame(l1c_dicts)
        df_l2a = pd.DataFrame(l2a_dicts)

        if df_l1c.empty or df_l2a.empty:
            logger.warning(f"Found {len(l1c_items)} L1C products and {len(l2a_items)} L2A products (after filtering)")
            return None, None

        logger.info(f"Found {len(l1c_items)} L1C products and {len(l2a_items)} L2A products (after filtering)")

        # Create unique ID keys for matching
        df_l1c['id_key'] = df_l1c['id'].apply(remove_last_segment_rsplit)
        df_l2a['id_key'] = df_l2a['id'].apply(remove_last_segment_rsplit)
        df_l2a['id_key'] = df_l2a['id_key'].str.replace('MSIL2A_', 'MSIL1C_')

        # Remove duplicates
        df_l1c = df_l1c.drop_duplicates(subset='id_key', keep='first')
        df_l2a = df_l2a.drop_duplicates(subset='id_key', keep='first')

        # Find matching items
        df_l2a = df_l2a[df_l2a['id_key'].isin(df_l1c['id_key'])]
        df_l1c = df_l1c[df_l1c['id_key'].isin(df_l2a['id_key'])]

        # Ensure order is aligned
        df_l2a = df_l2a.set_index('id_key')
        df_l1c = df_l1c.set_index('id_key')
        df_l2a = df_l2a.loc[df_l1c.index].reset_index()
        df_l1c = df_l1c.reset_index()

        logger.info(f"Found {len(df_l1c)} matching L1C/L2A pairs")

        if len(df_l1c) == 0:
            return None, None

        # Select a random pair with a fixed seed for reproducibility

        random_idx = np.random.randint(0, len(df_l1c))

        selected_l1c = df_l1c.iloc[random_idx]
        selected_l2a = df_l2a.iloc[random_idx]

        # Convert back to STAC items
        l1c_item = next((item for item in l1c_items if item.id == selected_l1c['id']), None)
        l2a_item = next((item for item in l2a_items if item.id == selected_l2a['id']), None)

        logger.success(f"Selected L1C: {l1c_item.id}, L2A: {l2a_item.id}")
        return l1c_item, l2a_item

    except Exception as e:
        logger.error(f"Error fetching Sentinel data: {e}")
        return None, None


def data_query_most_recent(catalog, bbox: list, max_cloud_cover: int, default_timedelta: int = 30):
    """
    Fetch both L1C and L2A products from CDSE STAC catalog and find the most recent matching pair.
    By default, searches for images from the last 30 days.

    Args:
        catalog: STAC catalog client
        bbox: Bounding box coordinates [west, south, east, north]
        max_cloud_cover: Maximum cloud cover percentage
        start_date: Start date in format "YYYY-MM-DD" (optional, defaults to 30 days ago)
        end_date: End date in format "YYYY-MM-DD" (optional, defaults to today)

    Returns:
        tuple: (most recent matched L1C item, most recent matched L2A item)
    """
    try:
        # Look up for the most recent date 
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        start_date = (datetime.date.today() - datetime.timedelta(days=default_timedelta)).strftime("%Y-%m-%d")
        
        logger.info(f"Using date range: {start_date} to {end_date}")
        
        # L1C products
        logger.info(f"Searching for L1C products from {start_date} to {end_date} in bbox {bbox}")
        l1c_items = catalog.search(
            collections=['sentinel-2-l1c'],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            max_items=1000,
            sortby=["-datetime"]  
        ).item_collection()

        # L2A products
        logger.info(f"Searching for L2A products from {start_date} to {end_date} in bbox {bbox}")
        l2a_items = catalog.search(
            collections=['sentinel-2-l2a'],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=1000,
            sortby=["-datetime"] 
        ).item_collection()

        # Filter L2A items - remove those with high nodata percentage
        l2a_items = [item for item in l2a_items if item.properties.get("statistics", {}).get('nodata', 100) < 5]

        # Convert to dataframes for easier matching
        l1c_dicts = [item.to_dict() for item in l1c_items]
        l2a_dicts = [item.to_dict() for item in l2a_items]

        df_l1c = pd.DataFrame(l1c_dicts)
        df_l2a = pd.DataFrame(l2a_dicts)

        if df_l1c.empty or df_l2a.empty:
            logger.warning(f"Found {len(l1c_items)} L1C products and {len(l2a_items)} L2A products (after filtering)")
            return None, None

        logger.info(f"Found {len(l1c_items)} L1C products and {len(l2a_items)} L2A products (after filtering)")

        # Create unique ID keys for matching
        df_l1c['id_key'] = df_l1c['id'].apply(remove_last_segment_rsplit)
        df_l2a['id_key'] = df_l2a['id'].apply(remove_last_segment_rsplit)
        df_l2a['id_key'] = df_l2a['id_key'].str.replace('MSIL2A_', 'MSIL1C_')

        # Extract datetime for sorting
        df_l1c['datetime'] = pd.to_datetime(df_l1c['properties'].apply(lambda x: x.get('datetime')))
        df_l2a['datetime'] = pd.to_datetime(df_l2a['properties'].apply(lambda x: x.get('datetime')))

        # Remove duplicates, keeping the most recent for each id_key
        df_l1c = df_l1c.sort_values('datetime', ascending=False).drop_duplicates(subset='id_key', keep='first')
        df_l2a = df_l2a.sort_values('datetime', ascending=False).drop_duplicates(subset='id_key', keep='first')

        # Find matching items
        df_l2a = df_l2a[df_l2a['id_key'].isin(df_l1c['id_key'])]
        df_l1c = df_l1c[df_l1c['id_key'].isin(df_l2a['id_key'])]

        # Ensure order is aligned
        df_l2a = df_l2a.set_index('id_key')
        df_l1c = df_l1c.set_index('id_key')
        df_l2a = df_l2a.loc[df_l1c.index].reset_index()
        df_l1c = df_l1c.reset_index()

        logger.info(f"Found {len(df_l1c)} matching L1C/L2A pairs")

        if len(df_l1c) == 0:
            logger.warning(f"There are no L1C scenes available for the selected region and specified time.")
            return None, None

        # Select the most recent pair based on datetime
        # Sort by datetime descending and take the first (most recent)
        df_combined = pd.merge(df_l1c, df_l2a, on='id_key', suffixes=('_l1c', '_l2a'))
        df_combined = df_combined.sort_values('datetime_l1c', ascending=False)
        
        most_recent_pair = df_combined.iloc[0]
        
        selected_l1c_id = most_recent_pair['id_l1c']
        selected_l2a_id = most_recent_pair['id_l2a']

        l1c_item = next((item for item in l1c_items if item.id == selected_l1c_id), None)
        l2a_item = next((item for item in l2a_items if item.id == selected_l2a_id), None)

        if l1c_item and l2a_item:
            logger.success(f"Selected most recent pair - L1C: {l1c_item.id} ({l1c_item.properties.get('datetime')}), L2A: {l2a_item.id} ({l2a_item.properties.get('datetime')})")
        else:
            logger.error("Failed to find corresponding STAC items for selected pair")
            return None, None

        return l1c_item, l2a_item

    except Exception as e:
        logger.error(f"Error fetching Sentinel data: {e}")
        return None, None