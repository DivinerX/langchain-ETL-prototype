"""
Data merging module using fuzzy matching to associate unstructured data with structured entities.
"""
import pandas as pd
import logging
from rapidfuzz import fuzz, process
from typing import List, Dict

logger = logging.getLogger(__name__)


def fuzzy_match_text_to_name(text: str, names: List[str], threshold: int = 60) -> tuple:
    """
    Use fuzzy matching to find the best matching name for a text.
    
    Args:
        text: Text to match
        names: List of names to match against
        threshold: Minimum similarity score (0-100)
        
    Returns:
        Tuple of (best_match, score) or (None, 0) if no match above threshold
    """
    if not text or not names:
        return None, 0
    
    # Extract potential product names from text (simple heuristic)
    # Look for capitalized words that might be product names
    words = text.split()
    potential_names = []
    
    for i, word in enumerate(words):
        # Check for sequences of capitalized words
        if word[0].isupper() and len(word) > 2:
            # Try to find multi-word product names
            name_candidate = word
            for j in range(i + 1, min(i + 3, len(words))):
                if words[j][0].isupper():
                    name_candidate += " " + words[j]
                else:
                    break
            potential_names.append(name_candidate)
    
    # If no potential names found, use the full text
    if not potential_names:
        potential_names = [text]
    
    # Find best match
    best_match = None
    best_score = 0
    
    for candidate in potential_names:
        result = process.extractOne(candidate, names, scorer=fuzz.partial_ratio)
        if result:
            match, score, index = result
            if score > best_score and score >= threshold:
                best_score = score
                best_match = match
    
    return best_match, best_score


def merge_data(structured_df: pd.DataFrame, unstructured_data: List[Dict]) -> pd.DataFrame:
    """
    Merge structured and unstructured data using fuzzy matching.
    Supports multiple reviews per product by creating one row per review.
    
    Args:
        structured_df: DataFrame with structured data
        unstructured_data: List of dictionaries with unstructured data
        
    Returns:
        Merged DataFrame with one row per review (products can appear multiple times)
    """
    # Get list of product names for matching
    product_names = structured_df['Name'].tolist()
    
    # List to store all merged records (one per review)
    merged_records = []
    
    # Track which reviews have been matched to avoid duplicates
    matched_review_ids = set()
    
    # First pass: Match reviews by product ID (exact match)
    for review in unstructured_data:
        review_id = review.get('id')
        review_text = review.get('text', '')
        
        if not review_text:
            continue
        
        # Try to match by ID first (if review has matching product ID)
        if review_id and review_id in structured_df['ID'].values:
            product_row = structured_df[structured_df['ID'] == review_id].iloc[0]
            
            # Create a record for this review
            record = {
                'ID': product_row['ID'],
                'Name': product_row['Name'],
                'Category': product_row['Category'],
                'Price': product_row['Price'],
                'review_text': review_text,
                'review_id': review_id
            }
            merged_records.append(record)
            matched_review_ids.add(review_id)
    
    # Second pass: Match remaining reviews using fuzzy matching
    for review in unstructured_data:
        review_id = review.get('id')
        review_text = review.get('text', '')
        
        if not review_text:
            continue
        
        # Skip if already matched by ID
        if review_id in matched_review_ids:
            continue
        
        # Use fuzzy matching to find product
        best_match, score = fuzzy_match_text_to_name(review_text, product_names)
        
        if best_match:
            # Find the product row
            product_row = structured_df[structured_df['Name'] == best_match].iloc[0]
            
            # Create a record for this review
            record = {
                'ID': product_row['ID'],
                'Name': product_row['Name'],
                'Category': product_row['Category'],
                'Price': product_row['Price'],
                'review_text': review_text,
                'review_id': review_id
            }
            merged_records.append(record)
            matched_review_ids.add(review_id)
    
    # Create DataFrame from merged records
    if merged_records:
        merged_df = pd.DataFrame(merged_records)
    else:
        # If no reviews matched, return empty DataFrame with correct columns
        merged_df = structured_df.copy()
        merged_df['review_text'] = ""
        merged_df['review_id'] = None
        return merged_df
    
    # Add products without reviews (optional - uncomment if you want to keep all products)
    products_with_reviews = set(merged_df['ID'].unique())
    products_without_reviews = structured_df[~structured_df['ID'].isin(products_with_reviews)].copy()
    if len(products_without_reviews) > 0:
        products_without_reviews['review_text'] = ""
        products_without_reviews['review_id'] = None
        merged_df = pd.concat([merged_df, products_without_reviews], ignore_index=True)
    
    # Count unique products and total reviews
    unique_products = merged_df['ID'].nunique()
    total_reviews = len(merged_df[merged_df['review_text'] != ''])
    
    logger.info(f"Merged data: {len(merged_df)} total records ({unique_products} unique products, {total_reviews} reviews)")
    return merged_df


if __name__ == "__main__":
    # Test merging
    from ingest import load_structured_data, load_unstructured_data
    from clean import clean_structured_data, clean_unstructured_data
    
    structured = load_structured_data()
    unstructured = load_unstructured_data()
    
    cleaned_structured = clean_structured_data(structured)
    cleaned_unstructured = clean_unstructured_data(unstructured)
    
    merged = merge_data(cleaned_structured, cleaned_unstructured)
    logger.info(f"\nMerged data shape: {merged.shape}")
    logger.info(f"\nSample merged record:")
    logger.info(merged[merged['review_text'] != ''].head(1).to_dict('records')[0] if len(merged[merged['review_text'] != '']) > 0 else "No matches found")

