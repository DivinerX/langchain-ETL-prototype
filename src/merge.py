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
            match, score = result
            if score > best_score and score >= threshold:
                best_score = score
                best_match = match
    
    return best_match, best_score


def merge_data(structured_df: pd.DataFrame, unstructured_data: List[Dict]) -> pd.DataFrame:
    """
    Merge structured and unstructured data using fuzzy matching.
    
    Args:
        structured_df: DataFrame with structured data
        unstructured_data: List of dictionaries with unstructured data
        
    Returns:
        Merged DataFrame
    """
    # Create a copy of structured data
    merged_df = structured_df.copy()
    
    # Initialize columns for unstructured data
    merged_df['review_text'] = ""
    merged_df['review_id'] = None
    
    # Get list of product names for matching
    product_names = structured_df['Name'].tolist()
    
    # Create a mapping of matched unstructured data
    matched_reviews = {}
    
    for review in unstructured_data:
        review_id = review.get('id')
        review_text = review.get('text', '')
        
        # Try to match by ID first (if review has matching ID)
        if review_id and review_id in structured_df['ID'].values:
            idx = structured_df[structured_df['ID'] == review_id].index[0]
            if idx not in matched_reviews:
                matched_reviews[idx] = {
                    'review_text': review_text,
                    'review_id': review_id
                }
            continue
        
        # Otherwise, use fuzzy matching
        best_match, score = fuzzy_match_text_to_name(review_text, product_names)
        
        if best_match:
            # Find the index of the matched product
            idx = structured_df[structured_df['Name'] == best_match].index[0]
            
            # Only assign if not already matched or if this match is better
            if idx not in matched_reviews:
                matched_reviews[idx] = {
                    'review_text': review_text,
                    'review_id': review_id
                }
    
    # Assign matched reviews to dataframe
    for idx, review_data in matched_reviews.items():
        merged_df.at[idx, 'review_text'] = review_data['review_text']
        merged_df.at[idx, 'review_id'] = review_data['review_id']
    
    # Remove rows without reviews (optional - comment out if you want to keep all products)
    # merged_df = merged_df[merged_df['review_text'] != ""]
    
    logger.info(f"Merged data: {len(merged_df)} records, {len(matched_reviews)} with reviews")
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

