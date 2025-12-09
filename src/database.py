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
    
    def insert_record(self, record: Dict) -> int:
        """
        Insert a single record into the database.
        
        Args:
            record: Dictionary containing record data
            
        Returns:
            ID of inserted record
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO enriched_records 
            (product_id, product_name, category, price, review_text, review_id, sentiment, topics, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.get('ID'),
            record.get('Name'),
            record.get('Category'),
            record.get('Price'),
            record.get('review_text', ''),
            record.get('review_id'),
            record.get('sentiment', ''),
            record.get('topics', ''),
            record.get('summary', '')
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_dataframe(self, df: pd.DataFrame):
        """
        Insert DataFrame records into the database.
        
        Args:
            df: DataFrame to insert
        """
        # Clear existing records (optional - comment out if you want to keep history)
        # self.clear_all_records()
        
        total = len(df)
        for idx, row in df.iterrows():
            record = row.to_dict()
            self.insert_record(record)
            
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                logger.info(f"Inserted {idx + 1}/{total} records...")
        
        logger.info(f"Database insert complete: {total} records")
    
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
    from ingest import load_structured_data, load_unstructured_data
    from clean import clean_structured_data, clean_unstructured_data
    from merge import merge_data
    
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

