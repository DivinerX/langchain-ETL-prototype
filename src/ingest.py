"""
Data ingestion module for loading structured and unstructured data sources.
"""
import pandas as pd
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_structured_data(file_path: str = "data/structured.csv") -> pd.DataFrame:
    """
    Load structured CSV data.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing structured data
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Structured data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} structured records from {file_path}")
    return df


def load_unstructured_data(file_path: str = "data/unstructured.json") -> list:
    """
    Load unstructured JSON data.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries containing unstructured data
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Unstructured data file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} unstructured records from {file_path}")
    return data


if __name__ == "__main__":
    # Test ingestion
    structured = load_structured_data()
    unstructured = load_unstructured_data()
    logger.info(f"\nStructured data shape: {structured.shape}")
    logger.info(f"Unstructured data count: {len(unstructured)}")

