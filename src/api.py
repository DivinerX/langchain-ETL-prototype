"""
FastAPI application for exposing enriched data through REST endpoints.
"""
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from src.database import Database

app = FastAPI(title="AI-Powered Data Enrichment API", version="1.0.0")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
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
            "GET /products/{product_id}/summary": "Get product summary with review statistics"
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
        # Replace NaN values with None to ensure Pydantic validation passes
        df = df.replace({np.nan: None})
        records = df.to_dict('records')
        return records
    except Exception as e:
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
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

