"""
Main pipeline orchestrator that runs the complete data enrichment workflow.
"""
import os
import sys
import pandas as pd
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to allow imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import all pipeline modules
from src.ingest import load_structured_data, load_unstructured_data
from src.clean import clean_structured_data, clean_unstructured_data
from src.merge import merge_data
from src.enrich_llm import enrich_dataframe
from src.embeddings import add_embeddings_to_dataframe
from src.database import Database
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException
from typing import List, Optional

logger = logging.getLogger(__name__)


def setup_pinecone() -> tuple[Optional[Pinecone], Optional[int]]:
    """
    Setup Pinecone connection for vector storage.
    
    Returns:
        Tuple of (Pinecone client or None, index dimension or None)
    """
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key or not index_name:
        logger.warning("Pinecone credentials not found. Skipping vector storage.")
        return None, None
    
    try:
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists, create if not
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=512,  # Default embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info(f"Index {index_name} created successfully")
            return pc, 512  # Return client and dimension
        else:
            # Get existing index dimension
            try:
                index = pc.Index(index_name)
                # Get dimension from index stats or description
                index_stats = index.describe_index_stats()
                # Try to get dimension from index info
                try:
                    index_info = pc.describe_index(index_name)
                    dimension = index_info.dimension
                except AttributeError:
                    # Fallback: try to get from index stats or use default
                    # If we can't get it, we'll need to check the first vector
                    dimension = None
                    logger.warning(f"Could not determine index dimension, will check from embeddings")
                
                if dimension:
                    logger.info(f"Using existing Pinecone index: {index_name} (dimension: {dimension})")
                else:
                    logger.info(f"Using existing Pinecone index: {index_name}")
                return pc, dimension
            except Exception as e:
                logger.error(f"Error getting index info: {e}")
                return pc, None
        
    except Exception as e:
        logger.error(f"Error setting up Pinecone: {e}")
        return None, None


def upsert_to_pinecone(pc: Pinecone, df, index_name: str, expected_dimension: Optional[int] = None):
    """
    Upsert embeddings to Pinecone.
    Clears the index before upserting to ensure a fresh start.
    
    Args:
        pc: Pinecone client
        df: DataFrame with embeddings
        index_name: Name of the Pinecone index
        expected_dimension: Expected dimension of the index (for validation)
    """
    if pc is None:
        return
    
    try:
        index = pc.Index(index_name)
        
        # Clear the index before upserting
        logger.info(f"Clearing Pinecone index: {index_name}")
        
        # Check index stats before delete
        try:
            stats_before = index.describe_index_stats()
            logger.info(f"Index stats before delete: {stats_before}")
        except Exception as e:
            logger.debug(f"Could not get index stats before delete: {e}")
        
        try:
            # Try to delete all vectors - use delete_all=True without namespace to delete from all namespaces
            # This is safer than specifying namespace="" which might cause issues
            logger.debug("Attempting delete_all=True without namespace parameter")
            try:
                index.delete(delete_all=True)
                logger.info("Deleted all vectors from all namespaces")
            except NotFoundException as e:
                # If namespace not found, index is already empty - this is fine
                if "Namespace not found" in str(e) or "namespace" in str(e).lower():
                    logger.info(f"Index {index_name} is already empty (no namespace found), proceeding with upsert")
                else:
                    logger.warning(f"NotFoundException during delete: {e}")
                    raise
            except Exception as delete_error:
                logger.warning(f"Delete operation error: {delete_error}, type: {type(delete_error).__name__}")
                # Try alternative delete method
                try:
                    logger.debug("Trying delete with namespace=''")
                    index.delete(delete_all=True, namespace="")
                    logger.info("Deleted all vectors from default namespace")
                except Exception as e2:
                    logger.warning(f"Alternative delete also failed: {e2}. Proceeding with upsert anyway.")
            
            # Small delay to ensure delete operation completes
            time.sleep(2)
            
            # Verify delete worked
            try:
                stats_after = index.describe_index_stats()
                total_after = stats_after.get('total_vector_count', 0)
                logger.info(f"Index stats after delete: {stats_after}")
                if total_after > 0:
                    logger.warning(f"WARNING: Index still has {total_after} vectors after delete operation")
                else:
                    logger.info(f"Index {index_name} cleared successfully (verified: 0 vectors)")
            except Exception as e:
                logger.debug(f"Could not verify delete: {e}")
                logger.info(f"Index {index_name} delete operation completed")
        except Exception as e:
            # Log other errors but don't fail - might be empty index
            logger.warning(f"Could not clear index (might be empty): {e}. Proceeding with upsert.")
        
        # Log dataframe info
        logger.info(f"Preparing to upsert vectors from DataFrame with {len(df)} rows")
        
        vectors_to_upsert = []
        skipped_count = 0
        for idx, row in df.iterrows():
            embedding = row.get('embedding')
            
            # Skip if embedding is None or invalid
            if embedding is None:
                skipped_count += 1
                continue
            
            # Check if embedding is valid (has length > 0)
            # This avoids the ambiguous truth value error with arrays
            try:
                if not hasattr(embedding, '__len__') or len(embedding) == 0:
                    skipped_count += 1
                    continue
            except (TypeError, ValueError):
                skipped_count += 1
                continue
            
            vector_id = f"record_{row.get('ID', idx)}"
            metadata = {
                'product_id': int(row.get('ID', 0)),
                'product_name': str(row.get('Name', '')),
                'category': str(row.get('Category', '')),
                'sentiment': str(row.get('sentiment', '')),
                'topics': str(row.get('topics', '')),
                'review_text': str(row.get('review_text', ''))[:1000]  # Limit metadata size
            }
            
            # Convert embedding to list if it's a numpy array or pandas Series
            if hasattr(embedding, 'tolist'):
                embedding_values = embedding.tolist()
            elif isinstance(embedding, (list, tuple)):
                embedding_values = list(embedding)
            else:
                embedding_values = embedding
            
            vectors_to_upsert.append({
                'id': vector_id,
                'values': embedding_values,
                'metadata': metadata
            })
        
        logger.info(f"Prepared {len(vectors_to_upsert)} vectors to upsert (skipped {skipped_count} invalid embeddings)")
        
        if vectors_to_upsert:
            # Upsert in batches
            batch_size = 100
            total_batches = (len(vectors_to_upsert) + batch_size - 1) // batch_size
            logger.info(f"Starting upsert of {len(vectors_to_upsert)} vectors in {total_batches} batch(es)")
            
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                batch_num = i//batch_size + 1
                try:
                    # Debug: Log first vector details
                    if batch_num == 1 and len(batch) > 0:
                        first_vector = batch[0]
                        logger.debug(f"First vector in batch: id={first_vector.get('id')}, values_len={len(first_vector.get('values', []))}, metadata_keys={list(first_vector.get('metadata', {}).keys())}")
                    
                    # Upsert to default namespace (empty string) - explicitly specify to ensure consistency
                    # If namespace parameter causes issues, it will fall back to default
                    upsert_response = None
                    upsert_success = False
                    try:
                        logger.debug(f"Attempting upsert with namespace='' for batch {batch_num}")
                        upsert_response = index.upsert(vectors=batch, namespace="")
                        upsert_success = True
                        logger.debug(f"Upsert with namespace='' succeeded")
                    except TypeError as te:
                        # If namespace parameter not supported, use without it (defaults to empty namespace)
                        logger.debug(f"Namespace parameter not supported, trying without namespace parameter: {te}")
                        try:
                            upsert_response = index.upsert(vectors=batch)
                            upsert_success = True
                            logger.debug(f"Upsert without namespace parameter succeeded")
                        except Exception as e2:
                            logger.error(f"Upsert failed even without namespace parameter: {e2}")
                            raise
                    except Exception as upsert_error:
                        logger.error(f"Upsert with namespace='' failed: {upsert_error}")
                        logger.error(f"Error type: {type(upsert_error).__name__}")
                        raise
                    
                    # Log detailed response
                    if upsert_response is not None:
                        logger.info(f"Upsert response type: {type(upsert_response)}")
                        logger.info(f"Upsert response: {upsert_response}")
                        if hasattr(upsert_response, 'upserted_count'):
                            logger.info(f"Upserted count from response: {upsert_response.upserted_count}")
                        if hasattr(upsert_response, '__dict__'):
                            logger.debug(f"Upsert response attributes: {upsert_response.__dict__}")
                    
                    logger.info(f"Upserted batch {batch_num}/{total_batches} to Pinecone ({len(batch)} vectors)")
                except Exception as batch_error:
                    logger.error(f"Error upserting batch {batch_num}: {batch_error}")
                    logger.error(f"Error details: {type(batch_error).__name__}: {str(batch_error)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise
            
            # Wait a moment for eventual consistency
            time.sleep(2)
            
            # Verify the upsert by checking index stats
            try:
                # Check stats for default namespace (empty string)
                stats = index.describe_index_stats()
                total_vectors = stats.get('total_vector_count', 0)
                
                # Also check namespace-specific stats if available
                namespaces = stats.get('namespaces', {})
                default_ns_count = namespaces.get('', {}).get('vector_count', 0) if namespaces else 0
                
                logger.info(f"Index verification: {total_vectors} total vectors in index")
                logger.info(f"Full index stats: {stats}")
                if namespaces:
                    logger.info(f"Namespace breakdown: {namespaces}")
                if default_ns_count > 0:
                    logger.info(f"Default namespace contains: {default_ns_count} vectors")
                
                # Try to query a vector to verify it's actually there
                if len(vectors_to_upsert) > 0:
                    test_vector_id = vectors_to_upsert[0].get('id')
                    logger.debug(f"Attempting to fetch test vector: {test_vector_id}")
                    try:
                        # Try fetching the vector
                        fetch_response = index.fetch(ids=[test_vector_id], namespace="")
                        logger.info(f"Test fetch response: {fetch_response}")
                        if test_vector_id in fetch_response.get('vectors', {}):
                            logger.info(f"SUCCESS: Test vector {test_vector_id} is retrievable from index")
                        else:
                            logger.warning(f"WARNING: Test vector {test_vector_id} not found in fetch response")
                    except Exception as fetch_error:
                        logger.warning(f"Could not fetch test vector: {fetch_error}")
                        # Try without namespace
                        try:
                            fetch_response = index.fetch(ids=[test_vector_id])
                            logger.info(f"Test fetch (no namespace) response: {fetch_response}")
                        except Exception as fetch_error2:
                            logger.warning(f"Could not fetch test vector without namespace: {fetch_error2}")
                
                if total_vectors == 0 and default_ns_count == 0:
                    logger.warning(f"WARNING: Index appears empty after upsert! Expected {len(vectors_to_upsert)} vectors.")
                    logger.warning("This might be due to namespace mismatch or eventual consistency delay.")
                elif total_vectors != len(vectors_to_upsert) and default_ns_count != len(vectors_to_upsert):
                    logger.warning(f"Vector count mismatch: Expected {len(vectors_to_upsert)}, found {total_vectors} total, {default_ns_count} in default namespace")
                else:
                    verified_count = default_ns_count if default_ns_count > 0 else total_vectors
                    logger.info(f"Successfully verified: {verified_count} vectors in index")
            except Exception as verify_error:
                logger.warning(f"Could not verify index stats: {verify_error}")
                import traceback
                logger.warning(f"Verification traceback: {traceback.format_exc()}")
            
            logger.info(f"Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone")
        else:
            logger.warning(f"No valid vectors to upsert. Check if embeddings are properly generated.")
    except Exception as e:
        error_msg = str(e)
        # Check if it's a dimension mismatch error
        if "dimension" in error_msg.lower() and ("does not match" in error_msg.lower() or "match" in error_msg.lower()):
            # Extract expected dimension from error message
            import re
            # Try multiple patterns to extract dimension
            match = re.search(r'dimension[^\d]*(\d+)', error_msg, re.IGNORECASE)
            if not match:
                match = re.search(r'(\d+)\s+does not match', error_msg, re.IGNORECASE)
            if not match:
                match = re.search(r'match the dimension of the index\s+(\d+)', error_msg, re.IGNORECASE)
            
            if match:
                expected_dim = int(match.group(1))
                logger.error(f"Dimension mismatch detected. Index expects {expected_dim} dimensions.")
                raise ValueError(f"Embedding dimension mismatch. Index requires {expected_dim} dimensions. "
                               f"Please regenerate embeddings with dimensions={expected_dim}.")
        logger.error(f"Error upserting to Pinecone: {e}")
        raise


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
    db_path = os.getenv("DB_PATH")
    if db_path:
        logger.info(f"Using database path from environment: {db_path}")
        db = Database(db_path=db_path)
    else:
        logger.info("Using default database path: enriched_data.db")
        db = Database()
    db.insert_dataframe(final_data)
    db.close()
    
    # Step 7: Upsert to Pinecone (if configured)
    logger.info("\n[7/7] Upserting embeddings to Pinecone...")
    pc, index_dimension = setup_pinecone()
    if pc:
        index_name = os.getenv("PINECONE_INDEX_NAME")
        records_with_embeddings = final_data[final_data['embedding'].notna()].copy()
        if len(records_with_embeddings) > 0:
            # Check if embeddings match index dimension
            sample_embedding = None
            for idx, row in records_with_embeddings.iterrows():
                emb = row.get('embedding')
                if emb is not None and hasattr(emb, '__len__') and len(emb) > 0:
                    sample_embedding = emb
                    break
            
            if sample_embedding is not None:
                embedding_dim = len(sample_embedding)
                
                # If we know the index dimension and they don't match, regenerate
                if index_dimension and embedding_dim != index_dimension:
                    logger.info(f"Embedding dimension ({embedding_dim}) doesn't match index dimension ({index_dimension})")
                    logger.info("Regenerating embeddings with correct dimension...")
                    from src.embeddings import EmbeddingGenerator
                    generator = EmbeddingGenerator(model="text-embedding-3-small")
                    for idx, row in records_with_embeddings.iterrows():
                        text = row.get('review_text', '')
                        if text and len(text.strip()) > 0:
                            embedding = generator.generate_embedding(text, dimensions=index_dimension)
                            records_with_embeddings.at[idx, 'embedding'] = embedding
                    logger.info("Embeddings regenerated with correct dimension")
                else:
                    # Try to upsert - will catch dimension mismatch if it occurs
                    try:
                        upsert_to_pinecone(pc, records_with_embeddings, index_name, expected_dimension=index_dimension)
                    except ValueError as ve:
                        # Dimension mismatch error - regenerate embeddings
                        error_msg = str(ve)
                        import re
                        # Try multiple patterns to match the error message
                        match = re.search(r'(\d+)\s+dimensions', error_msg)
                        if not match:
                            match = re.search(r'dimension\s+(\d+)', error_msg, re.IGNORECASE)
                        if not match:
                            match = re.search(r'(\d+)\s+does not match', error_msg)
                        if not match:
                            match = re.search(r'match the dimension of the index\s+(\d+)', error_msg, re.IGNORECASE)
                        if match:
                            expected_dim = int(match.group(1))
                            logger.info(f"Detected index dimension mismatch. Expected: {expected_dim}, Got: {embedding_dim}")
                            logger.info("Regenerating embeddings with correct dimension...")
                            from src.embeddings import EmbeddingGenerator
                            generator = EmbeddingGenerator(model="text-embedding-3-small")
                            for idx, row in records_with_embeddings.iterrows():
                                text = row.get('review_text', '')
                                if text and len(text.strip()) > 0:
                                    embedding = generator.generate_embedding(text, dimensions=expected_dim)
                                    records_with_embeddings.at[idx, 'embedding'] = embedding
                            logger.info("Embeddings regenerated with correct dimension")
                            # Retry upsert
                            upsert_to_pinecone(pc, records_with_embeddings, index_name, expected_dimension=expected_dim)
                        else:
                            raise
            else:
                logger.info("No valid embeddings found to upsert")
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

