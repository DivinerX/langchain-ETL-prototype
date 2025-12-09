"""
Main pipeline orchestrator that runs the complete data enrichment workflow.
"""
import os
import pandas as pd
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import all pipeline modules
from ingest import load_structured_data, load_unstructured_data
from clean import clean_structured_data, clean_unstructured_data
from merge import merge_data
from enrich_llm import enrich_dataframe
from embeddings import add_embeddings_to_dataframe
from database import Database
from pinecone import Pinecone, ServerlessSpec
from typing import List, Optional

logger = logging.getLogger(__name__)


def setup_pinecone() -> Optional[Pinecone]:
    """
    Setup Pinecone connection for vector storage.
    
    Returns:
        Pinecone client or None if not configured
    """
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key or not index_name:
        logger.warning("Pinecone credentials not found. Skipping vector storage.")
        return None
    
    try:
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists, create if not
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI ada-002 embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info(f"Index {index_name} created successfully")
        else:
            logger.info(f"Using existing Pinecone index: {index_name}")
        
        return pc
    except Exception as e:
        logger.error(f"Error setting up Pinecone: {e}")
        return None


def upsert_to_pinecone(pc: Pinecone, df, index_name: str):
    """
    Upsert embeddings to Pinecone.
    
    Args:
        pc: Pinecone client
        df: DataFrame with embeddings
        index_name: Name of the Pinecone index
    """
    if pc is None:
        return
    
    try:
        index = pc.Index(index_name)
        
        vectors_to_upsert = []
        for idx, row in df.iterrows():
            if row.get('embedding') is not None and pd.notna(row.get('embedding')):
                vector_id = f"record_{row.get('ID', idx)}"
                metadata = {
                    'product_id': int(row.get('ID', 0)),
                    'product_name': str(row.get('Name', '')),
                    'category': str(row.get('Category', '')),
                    'sentiment': str(row.get('sentiment', '')),
                    'topics': str(row.get('topics', '')),
                    'review_text': str(row.get('review_text', ''))[:1000]  # Limit metadata size
                }
                
                vectors_to_upsert.append({
                    'id': vector_id,
                    'values': row['embedding'],
                    'metadata': metadata
                })
        
        if vectors_to_upsert:
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1} to Pinecone ({len(batch)} vectors)")
            
            logger.info(f"Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone")
    except Exception as e:
        logger.error(f"Error upserting to Pinecone: {e}")


def run_pipeline():
    """Run the complete data enrichment pipeline."""
    logger.info("=" * 60)
    logger.info("Starting AI-Powered Data Enrichment Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Ingestion
    logger.info("\n[1/7] Ingesting data...")
    structured_data = load_structured_data()
    unstructured_data = load_unstructured_data()
    
    # Step 2: Cleaning
    logger.info("\n[2/7] Cleaning and normalizing data...")
    cleaned_structured = clean_structured_data(structured_data)
    cleaned_unstructured = clean_unstructured_data(unstructured_data)
    
    # Step 3: Merging
    logger.info("\n[3/7] Merging structured and unstructured data...")
    merged_data = merge_data(cleaned_structured, cleaned_unstructured)
    
    # Step 4: LLM Enrichment
    logger.info("\n[4/7] Enriching data with LLM...")
    records_with_reviews = merged_data[merged_data['review_text'] != ''].copy()
    
    if len(records_with_reviews) > 0:
        enriched_data = enrich_dataframe(records_with_reviews, text_column='review_text')
        
        # Merge enriched data back with records without reviews
        records_without_reviews = merged_data[merged_data['review_text'] == ''].copy()
        records_without_reviews['sentiment'] = ''
        records_without_reviews['topics'] = ''
        records_without_reviews['summary'] = ''
        
        final_data = pd.concat([enriched_data, records_without_reviews], ignore_index=True)
    else:
        logger.info("No records with reviews to enrich")
        final_data = merged_data.copy()
        final_data['sentiment'] = ''
        final_data['topics'] = ''
        final_data['summary'] = ''
    
    # Step 5: Generate Embeddings
    logger.info("\n[5/7] Generating embeddings...")
    records_with_reviews = final_data[final_data['review_text'] != ''].copy()
    
    if len(records_with_reviews) > 0:
        records_with_embeddings = add_embeddings_to_dataframe(records_with_reviews)
        
        # Merge back
        records_without_reviews = final_data[final_data['review_text'] == ''].copy()
        records_without_reviews['embedding'] = None
        
        final_data = pd.concat([records_with_embeddings, records_without_reviews], ignore_index=True)
    else:
        final_data['embedding'] = None
    
    # Step 6: Store in SQLite
    logger.info("\n[6/7] Storing data in SQLite database...")
    db = Database()
    db.insert_dataframe(final_data)
    db.close()
    
    # Step 7: Upsert to Pinecone (if configured)
    logger.info("\n[7/7] Upserting embeddings to Pinecone...")
    pc = setup_pinecone()
    if pc:
        index_name = os.getenv("PINECONE_INDEX_NAME")
        records_with_embeddings = final_data[final_data['embedding'].notna()].copy()
        if len(records_with_embeddings) > 0:
            upsert_to_pinecone(pc, records_with_embeddings, index_name)
        else:
            logger.info("No embeddings to upsert")
    else:
        logger.info("Pinecone not configured, skipping vector storage")
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)
    logger.info(f"\nSummary:")
    logger.info(f"  - Total records: {len(final_data)}")
    logger.info(f"  - Records with reviews: {len(final_data[final_data['review_text'] != ''])}")
    logger.info(f"  - Records with embeddings: {len(final_data[final_data['embedding'].notna()])}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Start the API: uvicorn src.api:app --reload")
    logger.info(f"  2. Launch the frontend: streamlit run streamlit_app.py")


if __name__ == "__main__":
    run_pipeline()

