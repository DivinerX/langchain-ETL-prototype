"""
FastAPI application for exposing enriched data through REST endpoints.
"""
import os
import numpy as np
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from src.database import Database

logger = logging.getLogger(__name__)

app = FastAPI(title="AI-Powered Data Enrichment API", version="1.0.0")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database - use environment variable if set
db_path = os.getenv("DB_PATH")
if db_path:
    # Validate the path - reject Windows paths on Linux
    if len(db_path) > 2 and db_path[1] == ':' and os.path.sep == '/':
        logger.warning(f"Invalid Windows path detected in DB_PATH: {db_path}. Using default instead.")
        db_path = None

if db_path:
    logger.info(f"Using database path from environment: {db_path}")
    db = Database(db_path=db_path)
else:
    logger.info("Using default database path (will be resolved relative to working directory)")
    db = Database()


class RecordResponse(BaseModel):
    """Response model for a single record."""
    id: int
    product_id: Optional[int]
    product_name: Optional[str]
    category: Optional[str]
    price: Optional[float]
    review_text: Optional[str]
    review_id: Optional[int]
    sentiment: Optional[str]
    topics: Optional[str]
    summary: Optional[str]
    
    class Config:
        from_attributes = True


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI-Powered Data Enrichment API",
        "version": "1.0.0",
        "endpoints": {
            "GET /records": "Get all enriched records",
            "GET /records/{id}": "Get a specific record by ID",
            "GET /search": "Search records by query (semantic search)",
            "GET /products/{product_id}/reviews": "Get all reviews for a specific product",
            "GET /products/{product_id}/summary": "Get product summary with review statistics",
            "GET /health": "Health check with database status",
            "GET /debug/database": "Debug endpoint to inspect database contents"
        }
    }


@app.get("/records", response_model=List[RecordResponse])
async def get_all_records():
    """
    Get all enriched records.
    
    Returns:
        List of all enriched records
    """
    try:
        df = db.get_all_records()
        logger.info(f"Retrieved {len(df)} records from database")
        # Replace NaN values with None to ensure Pydantic validation passes
        df = df.replace({np.nan: None})
        records = df.to_dict('records')
        return records
    except Exception as e:
        logger.error(f"Error retrieving records: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving records: {str(e)}")


@app.get("/records/{record_id}", response_model=RecordResponse)
async def get_record_by_id(record_id: int):
    """
    Get a specific record by ID.
    
    Args:
        record_id: ID of the record to retrieve
        
    Returns:
        Record data
    """
    try:
        record = db.get_record_by_id(record_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Record with ID {record_id} not found")
        # Replace any NaN values with None
        record = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in record.items()}
        return record
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving record: {str(e)}")


@app.get("/search")
async def search_records(
    query: Optional[str] = None,
    sentiment: Optional[str] = None,
    topic: Optional[str] = None
):
    """
    Search records by query, sentiment, or topic.
    
    Args:
        query: Search query (for semantic search - requires Pinecone)
        sentiment: Filter by sentiment (positive, negative, neutral)
        topic: Filter by topic (partial match)
        
    Returns:
        List of matching records
    """
    try:
        if sentiment:
            df = db.search_by_sentiment(sentiment)
        elif topic:
            df = db.search_by_topic(topic)
        elif query:
            # For semantic search, we would use Pinecone here
            # For now, return a simple text search
            all_records = db.get_all_records()
            # Simple text matching (in production, use Pinecone for semantic search)
            df = all_records[
                all_records['review_text'].str.contains(query, case=False, na=False) |
                all_records['product_name'].str.contains(query, case=False, na=False)
            ]
        else:
            df = db.get_all_records()
        
        # Replace NaN values with None to ensure Pydantic validation passes
        df = df.replace({np.nan: None})
        records = df.to_dict('records')
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching records: {str(e)}")


@app.get("/products/{product_id}/reviews", response_model=List[RecordResponse])
async def get_product_reviews(product_id: int):
    """
    Get all reviews for a specific product.
    
    Args:
        product_id: ID of the product to get reviews for
        
    Returns:
        List of all reviews for the product
    """
    try:
        df = db.get_reviews_by_product_id(product_id)
        if len(df) == 0:
            raise HTTPException(status_code=404, detail=f"No reviews found for product ID {product_id}")
        
        # Replace NaN values with None to ensure Pydantic validation passes
        df = df.replace({np.nan: None})
        records = df.to_dict('records')
        return records
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving reviews: {str(e)}")


@app.get("/products/{product_id}/summary")
async def get_product_summary(product_id: int):
    """
    Get product summary with aggregated review statistics.
    
    Args:
        product_id: ID of the product to get summary for
        
    Returns:
        Dictionary with product info and review statistics
    """
    try:
        summary = db.get_product_summary(product_id)
        if not summary:
            raise HTTPException(status_code=404, detail=f"Product with ID {product_id} not found")
        return summary
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving product summary: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint with database status."""
    try:
        # Check database connection and record count
        df = db.get_all_records()
        record_count = len(df)
        
        # Get database file size
        db_size = 0
        if os.path.exists(db.db_path):
            db_size = os.path.getsize(db.db_path)
        
        return {
            "status": "healthy",
            "database": {
                "path": db.db_path,
                "exists": os.path.exists(db.db_path),
                "file_size_bytes": db_size,
                "record_count": record_count
            },
            "environment": {
                "DB_PATH_env": os.getenv("DB_PATH", "not set"),
                "working_directory": os.getcwd(),
                "path_separator": os.path.sep
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "database": {
                "path": db.db_path if hasattr(db, 'db_path') else "unknown",
                "exists": os.path.exists(db.db_path) if hasattr(db, 'db_path') else False
            },
            "environment": {
                "DB_PATH_env": os.getenv("DB_PATH", "not set"),
                "working_directory": os.getcwd(),
                "path_separator": os.path.sep
            }
        }


@app.get("/debug/database")
async def debug_database():
    """Debug endpoint to inspect database contents and structure."""
    try:
        import sqlite3
        
        debug_info = {
            "database_path": db.db_path,
            "file_exists": os.path.exists(db.db_path),
            "file_size_bytes": os.path.getsize(db.db_path) if os.path.exists(db.db_path) else 0,
        }
        
        if not os.path.exists(db.db_path):
            return {
                **debug_info,
                "error": "Database file does not exist",
                "solution": "Run the data pipeline: fly ssh console -C 'python src/run_pipeline.py'"
            }
        
        # Connect directly to inspect
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        debug_info["tables"] = tables
        
        # Check if enriched_records table exists
        if "enriched_records" in tables:
            # Get table schema
            cursor.execute("PRAGMA table_info(enriched_records);")
            columns = [{"name": row[1], "type": row[2]} for row in cursor.fetchall()]
            debug_info["enriched_records_schema"] = columns
            
            # Get record count
            cursor.execute("SELECT COUNT(*) FROM enriched_records;")
            record_count = cursor.fetchone()[0]
            debug_info["enriched_records_count"] = record_count
            
            # Get sample records (first 3)
            if record_count > 0:
                cursor.execute("SELECT id, product_id, product_name, sentiment FROM enriched_records LIMIT 3;")
                sample_records = []
                for row in cursor.fetchall():
                    sample_records.append({
                        "id": row[0],
                        "product_id": row[1],
                        "product_name": row[2],
                        "sentiment": row[3]
                    })
                debug_info["sample_records"] = sample_records
            else:
                debug_info["sample_records"] = []
                debug_info["warning"] = "Table exists but is empty. Run the data pipeline to populate it."
                debug_info["solution"] = "Run: fly ssh console -C 'python src/run_pipeline.py'"
        else:
            debug_info["error"] = "enriched_records table does not exist"
            debug_info["solution"] = "Run the data pipeline: fly ssh console -C 'python src/run_pipeline.py'"
        
        conn.close()
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Debug database failed: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "database_path": db.db_path,
            "file_exists": os.path.exists(db.db_path) if hasattr(db, 'db_path') else False
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

