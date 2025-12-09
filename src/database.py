"""
Database module for SQLite operations to store enriched data.
"""
import sqlite3
import pandas as pd
import logging
import os
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Database:
    """Class for managing SQLite database operations."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file. If None, uses DB_PATH env var or defaults to "enriched_data.db"
        """
        # Get database path from parameter, environment variable, or default
        if db_path is None:
            db_path = os.getenv("DB_PATH")
            if not db_path:
                # Default to a path in the current working directory
                # On Fly.io/Docker, working directory is /app (from Dockerfile WORKDIR)
                # For local development, this will be the project root
                cwd = os.getcwd()
                # Use absolute path to avoid path resolution issues
                db_path = os.path.join(cwd, "enriched_data.db")
        
        # Clean up the path - handle Windows paths that might have leaked in
        # Remove any Windows-style absolute paths (C:, D:, etc.) when running on Linux
        # Check for Windows drive letters (C:, D:, etc.) anywhere in the path
        if db_path and os.path.sep == '/':  # Running on Linux/Unix
            # Check if path contains Windows drive letter pattern (X:)
            import re
            if re.search(r'[A-Za-z]:[/\\]', db_path):
                # This is a Windows path on a Linux system - ignore it and use default
                logger.warning(f"Detected Windows path '{db_path}' on Linux system. Using default path instead.")
                cwd = os.getcwd()
                db_path = os.path.join(cwd, "enriched_data.db")
        
        # Normalize the path properly
        db_path = os.path.normpath(db_path)
        
        # Check if it's an absolute path (Unix-style starting with /)
        # On Linux/Unix, absolute paths start with /
        if os.path.isabs(db_path) and db_path.startswith('/'):
            # Absolute path - use as is (e.g., /data/enriched_data.db or /enriched_data.db)
            self.db_path = db_path
        else:
            # Relative path - resolve to absolute path relative to current working directory
            # On Fly.io, working directory is /app (from Dockerfile WORKDIR)
            self.db_path = os.path.abspath(db_path)
        
        self.conn = None
        
        # Log the database path being used for debugging
        logger.info(f"Initializing database at path: {self.db_path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"DB_PATH environment variable: {os.getenv('DB_PATH', 'not set')}")
        logger.info(f"Database file exists: {os.path.exists(self.db_path)}")
        
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        # Ensure parent directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")
        
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
        
        # Check if there are existing records
        cursor.execute("SELECT COUNT(*) FROM enriched_records")
        record_count = cursor.fetchone()[0]
        logger.info(f"Database initialized: {self.db_path} (existing records: {record_count})")
    
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
    
    def get_reviews_by_product_id(self, product_id: int) -> pd.DataFrame:
        """
        Get all reviews for a specific product.
        
        Args:
            product_id: Product ID to get reviews for
            
        Returns:
            DataFrame with all reviews for the product
        """
        query = "SELECT * FROM enriched_records WHERE product_id = ? AND review_text != '' AND review_text IS NOT NULL"
        df = pd.read_sql_query(query, self.conn, params=(product_id,))
        return df
    
    def get_product_summary(self, product_id: int) -> Optional[Dict]:
        """
        Get product information with aggregated review statistics.
        
        Args:
            product_id: Product ID to get summary for
            
        Returns:
            Dictionary with product info and review statistics, or None if not found
        """
        # Get product info (from first record with this product_id)
        query = """
            SELECT product_id, product_name, category, price, 
                   COUNT(*) as review_count,
                   SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive_count,
                   SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative_count,
                   SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_count
            FROM enriched_records 
            WHERE product_id = ?
            GROUP BY product_id, product_name, category, price
        """
        df = pd.read_sql_query(query, self.conn, params=(product_id,))
        
        if len(df) == 0:
            return None
        
        row = df.iloc[0]
        return {
            'product_id': int(row['product_id']),
            'product_name': row['product_name'],
            'category': row['category'],
            'price': float(row['price']) if pd.notna(row['price']) else None,
            'review_count': int(row['review_count']),
            'positive_count': int(row['positive_count']),
            'negative_count': int(row['negative_count']),
            'neutral_count': int(row['neutral_count'])
        }
    
    def clear_all_records(self):
        """Clear all records from the database."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM enriched_records")
        self.conn.commit()
        logger.info("All records cleared from database")
    
    def get_table_info(self) -> Dict:
        """
        Get information about the database tables.
        
        Returns:
            Dictionary with table information
        """
        cursor = self.conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        info = {
            "tables": tables,
            "enriched_records_exists": "enriched_records" in tables
        }
        
        if "enriched_records" in tables:
            # Get record count
            cursor.execute("SELECT COUNT(*) FROM enriched_records;")
            info["enriched_records_count"] = cursor.fetchone()[0]
            
            # Get table schema
            cursor.execute("PRAGMA table_info(enriched_records);")
            info["enriched_records_schema"] = [
                {"name": row[1], "type": row[2], "notnull": row[3], "default": row[4]}
                for row in cursor.fetchall()
            ]
        
        return info
    
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

