"""
Database module for SQLite operations to store enriched data.
"""
import sqlite3
import pandas as pd
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Database:
    """Class for managing SQLite database operations."""
    
    def __init__(self, db_path: str = "enriched_data.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create enriched_records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enriched_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                product_name TEXT,
                category TEXT,
                price REAL,
                review_text TEXT,
                review_id INTEGER,
                sentiment TEXT,
                topics TEXT,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
        logger.info(f"Database initialized: {self.db_path}")
    
    def product_exists(self, product_id: int, review_id: Optional[int] = None) -> bool:
        """
        Check if a product (or product-review combination) already exists in the database.
        
        Args:
            product_id: Product ID to check
            review_id: Optional review ID to check (if provided, checks for product_id + review_id combination)
            
        Returns:
            True if product exists, False otherwise
        """
        cursor = self.conn.cursor()
        
        if review_id is not None:
            # Check for product_id + review_id combination
            cursor.execute("""
                SELECT COUNT(*) FROM enriched_records 
                WHERE product_id = ? AND review_id = ?
            """, (product_id, review_id))
        else:
            # Check for product_id only
            cursor.execute("""
                SELECT COUNT(*) FROM enriched_records 
                WHERE product_id = ?
            """, (product_id,))
        
        count = cursor.fetchone()[0]
        return count > 0
    
    def insert_record(self, record: Dict) -> Optional[int]:
        """
        Insert a single record into the database.
        Checks for duplicates before inserting based on product_id (and review_id if present).
        
        Args:
            record: Dictionary containing record data
            
        Returns:
            ID of inserted record, or None if record already exists (duplicate)
        """
        product_id = record.get('ID')
        review_id = record.get('review_id')
        
        # Check if product already exists
        if self.product_exists(product_id, review_id):
            logger.debug(f"Skipping duplicate record: product_id={product_id}, review_id={review_id}")
            return None
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO enriched_records 
            (product_id, product_name, category, price, review_text, review_id, sentiment, topics, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            product_id,
            record.get('Name'),
            record.get('Category'),
            record.get('Price'),
            record.get('review_text', ''),
            review_id,
            record.get('sentiment', ''),
            record.get('topics', ''),
            record.get('summary', '')
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_dataframe(self, df: pd.DataFrame):
        """
        Insert DataFrame records into the database.
        Skips duplicate records based on product_id (and review_id if present).
        
        Args:
            df: DataFrame to insert
        """
        # Clear existing records (optional - comment out if you want to keep history)
        # self.clear_all_records()
        
        total = len(df)
        inserted_count = 0
        skipped_count = 0
        
        for idx, row in df.iterrows():
            record = row.to_dict()
            result = self.insert_record(record)
            
            if result is not None:
                inserted_count += 1
            else:
                skipped_count += 1
            
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                logger.info(f"Processed {idx + 1}/{total} records (Inserted: {inserted_count}, Skipped: {skipped_count})...")
        
        logger.info(f"Database insert complete: {inserted_count} records inserted, {skipped_count} duplicates skipped")
    
    def get_all_records(self) -> pd.DataFrame:
        """
        Retrieve all records from the database.
        
        Returns:
            DataFrame with all records
        """
        query = "SELECT * FROM enriched_records"
        df = pd.read_sql_query(query, self.conn)
        return df
    
    def get_record_by_id(self, record_id: int) -> Optional[Dict]:
        """
        Retrieve a single record by ID.
        
        Args:
            record_id: ID of the record
            
        Returns:
            Dictionary with record data, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM enriched_records WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def search_by_sentiment(self, sentiment: str) -> pd.DataFrame:
        """
        Search records by sentiment.
        
        Args:
            sentiment: Sentiment to filter by (positive, negative, neutral)
            
        Returns:
            DataFrame with matching records
        """
        query = "SELECT * FROM enriched_records WHERE sentiment = ?"
        df = pd.read_sql_query(query, self.conn, params=(sentiment,))
        return df
    
    def search_by_topic(self, topic: str) -> pd.DataFrame:
        """
        Search records by topic (partial match).
        
        Args:
            topic: Topic to search for
            
        Returns:
            DataFrame with matching records
        """
        query = "SELECT * FROM enriched_records WHERE topics LIKE ?"
        df = pd.read_sql_query(query, self.conn, params=(f'%{topic}%',))
        return df
    
    def clear_all_records(self):
        """Clear all records from the database."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM enriched_records")
        self.conn.commit()
        logger.info("All records cleared from database")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    # Test database operations
    from src.ingest import load_structured_data, load_unstructured_data
    from src.clean import clean_structured_data, clean_unstructured_data
    from src.merge import merge_data
    
    structured = load_structured_data()
    unstructured = load_unstructured_data()
    
    cleaned_structured = clean_structured_data(structured)
    cleaned_unstructured = clean_unstructured_data(unstructured)
    
    merged = merge_data(cleaned_structured, cleaned_unstructured)
    
    # Test database
    db = Database()
    db.insert_dataframe(merged)
    
    # Retrieve records
    all_records = db.get_all_records()
    logger.info(f"\nRetrieved {len(all_records)} records from database")
    logger.info(all_records.head())
    
    db.close()

