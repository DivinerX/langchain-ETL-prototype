"""
Data cleaning and normalization module.
"""
import pandas as pd
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and standardizing format.
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text string
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_categorical(value: str) -> str:
    """
    Normalize categorical values.
    
    Args:
        value: Categorical value to normalize
        
    Returns:
        Normalized categorical value
    """
    if pd.isna(value) or value is None:
        return ""
    
    value = str(value).strip()
    # Capitalize first letter of each word
    value = value.title()
    
    return value


def clean_structured_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize structured data.
    
    Args:
        df: DataFrame with structured data
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    removed = initial_count - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate records")
    
    # Remove rows with missing IDs
    df = df.dropna(subset=['ID'])
    
    # Normalize text columns
    text_columns = ['Name', 'Category']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(normalize_text)
            df[col] = df[col].apply(normalize_categorical)
    
    # Ensure Price is numeric
    if 'Price' in df.columns:
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Price'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    logger.info(f"Cleaned structured data: {len(df)} records remaining")
    return df


def clean_unstructured_data(data: List[Dict]) -> List[Dict]:
    """
    Clean and normalize unstructured data.
    
    Args:
        data: List of dictionaries with unstructured data
        
    Returns:
        Cleaned list of dictionaries
    """
    cleaned = []
    seen_ids = set()
    
    for item in data:
        # Ensure required fields exist
        if 'id' not in item or 'text' not in item:
            continue
        
        # Remove duplicates based on ID
        if item['id'] in seen_ids:
            continue
        seen_ids.add(item['id'])
        
        # Normalize text
        item['text'] = normalize_text(item['text'])
        
        # Skip empty text
        if not item['text']:
            continue
        
        cleaned.append(item)
    
    logger.info(f"Cleaned unstructured data: {len(cleaned)} records remaining")
    return cleaned


if __name__ == "__main__":
    # Test cleaning
    from ingest import load_structured_data, load_unstructured_data
    
    structured = load_structured_data()
    unstructured = load_unstructured_data()
    
    cleaned_structured = clean_structured_data(structured)
    cleaned_unstructured = clean_unstructured_data(unstructured)
    
    logger.info(f"\nCleaned structured: {len(cleaned_structured)} records")
    logger.info(f"Cleaned unstructured: {len(cleaned_unstructured)} records")

