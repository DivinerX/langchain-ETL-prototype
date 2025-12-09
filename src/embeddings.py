"""
Embedding generation module for creating vector embeddings for semantic search.
"""
import os
import logging
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Class for generating embeddings using OpenAI."""
    
    def __init__(self, model: str = "text-embedding-3-small", dimensions: Optional[int] = 512):
        """
        Initialize the embedding generator.
        
        Args:
            model: OpenAI embedding model name (default: text-embedding-3-small)
            dimensions: Dimension size (default: 512, only supported for text-embedding-3 models)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
    
    def generate_embedding(self, text: str, dimensions: Optional[int] = None) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            dimensions: Optional dimension size (overrides instance dimensions if provided)
            
        Returns:
            List of floats representing the embedding, or None if error
        """
        if not text or len(text.strip()) == 0:
            return None
        
        try:
            # Use provided dimensions or instance dimensions
            dims = dimensions if dimensions is not None else self.dimensions
            
            # Build request parameters
            params = {
                "model": self.model,
                "input": text
            }
            
            # Add dimensions parameter if using text-embedding-3 models
            if dims is not None and "text-embedding-3" in self.model:
                params["dimensions"] = dims
            
            response = self.client.embeddings.create(**params)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings (or None for failed generations)
        """
        embeddings = []
        for text in texts:
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings


def add_embeddings_to_dataframe(df, text_column: str = 'review_text', dimensions: int = 512):
    """
    Add embeddings column to DataFrame.
    
    Args:
        df: DataFrame to add embeddings to
        text_column: Name of the column containing text
        dimensions: Embedding dimension size (default: 512)
        
    Returns:
        DataFrame with embeddings column
    """
    generator = EmbeddingGenerator(dimensions=dimensions)
    
    df = df.copy()
    df['embedding'] = None
    
    total = len(df)
    for idx, row in df.iterrows():
        text = row.get(text_column, "")
        
        if text and len(text.strip()) > 0:
            embedding = generator.generate_embedding(text)
            df.at[idx, 'embedding'] = embedding
        
        if (idx + 1) % 5 == 0 or (idx + 1) == total:
            logger.info(f"Generated embeddings for {idx + 1}/{total} records...")
    
    logger.info(f"Embedding generation complete: {total} records processed")
    return df


if __name__ == "__main__":
    # Test embedding generation
    from ingest import load_structured_data, load_unstructured_data
    from clean import clean_structured_data, clean_unstructured_data
    from merge import merge_data
    
    structured = load_structured_data()
    unstructured = load_unstructured_data()
    
    cleaned_structured = clean_structured_data(structured)
    cleaned_unstructured = clean_unstructured_data(unstructured)
    
    merged = merge_data(cleaned_structured, cleaned_unstructured)
    
    records_with_reviews = merged[merged['review_text'] != ''].copy()
    
    if len(records_with_reviews) > 0:
        logger.info(f"\nGenerating embeddings for {len(records_with_reviews)} records...")
        with_embeddings = add_embeddings_to_dataframe(records_with_reviews)
        logger.info(f"\nEmbeddings generated: {with_embeddings['embedding'].notna().sum()} records")
    else:
        logger.info("No records with reviews to generate embeddings for")

