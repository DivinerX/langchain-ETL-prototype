"""
LLM enrichment module using OpenAI and LangChain for sentiment analysis, topic extraction, and summarization.
"""
import os
import logging
import time
import random
from typing import Dict, Optional, Callable, Any
from functools import wraps
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
try:
    from openai import RateLimitError, APIError
except ImportError:
    # Fallback for older versions
    RateLimitError = Exception
    APIError = Exception
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Decorator that retries a function with exponential backoff on rate limit errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if it's a rate limit error
                    error_str = str(e).lower()
                    error_type = type(e).__name__.lower()
                    
                    # Check for rate limit indicators
                    is_rate_limit = (
                        isinstance(e, RateLimitError) if RateLimitError != Exception else False or
                        "429" in error_str or
                        "rate limit" in error_str or
                        "rate_limit" in error_str or
                        "ratelimit" in error_str or
                        "too many requests" in error_str or
                        "ratelimit" in error_type or
                        (hasattr(e, 'status_code') and e.status_code == 429) or
                        (hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 429)
                    )
                    
                    if not is_rate_limit:
                        # Not a rate limit error, re-raise immediately
                        raise
                    
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    if jitter:
                        # Add random jitter (0-25% of delay)
                        jitter_amount = delay * 0.25 * random.random()
                        sleep_time = min(delay + jitter_amount, max_delay)
                    else:
                        sleep_time = min(delay, max_delay)
                    
                    logger.warning(
                        f"Rate limit error in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {sleep_time:.2f} seconds... Error: {str(e)[:100]}"
                    )
                    
                    time.sleep(sleep_time)
                    delay *= exponential_base
                    last_exception = e
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class LLMEnricher:
    """Class for enriching data using LLM via LangChain."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize the LLM enricher.
        
        Args:
            model_name: OpenAI model name
            temperature: Temperature for LLM responses
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )
        # Set API key via environment variable (already loaded)
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize chains
        self._setup_chains()
    
    def _setup_chains(self):
        """Setup LangChain prompts for different enrichment tasks."""
        
        # Sentiment analysis prompt
        self.sentiment_prompt = ChatPromptTemplate.from_template(
            "Analyze the sentiment of the following text and respond with ONLY one word: "
            "positive, negative, or neutral.\n\nText: {text}\n\nSentiment:"
        )
        
        # Topic extraction prompt
        self.topic_prompt = ChatPromptTemplate.from_template(
            "Extract the main topics or themes from the following text. "
            "Respond with 2-3 comma-separated topics. Be concise.\n\nText: {text}\n\nTopics:"
        )
        
        # Summarization prompt
        self.summary_prompt = ChatPromptTemplate.from_template(
            "Provide a one-sentence summary of the following text. "
            "Be concise and informative.\n\nText: {text}\n\nSummary:"
        )
    
    @retry_with_exponential_backoff(max_retries=5, initial_delay=2.0, max_delay=120.0)
    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment label (positive, negative, or neutral)
        """
        if not text or len(text.strip()) == 0:
            return "neutral"
        
        try:
            # Use prompt + LLM directly (LangChain v1.x style)
            chain = self.sentiment_prompt | self.llm
            result = chain.invoke({"text": text})
            
            # Handle AIMessage object (LangChain v1.x response format)
            if hasattr(result, 'content'):
                sentiment = str(result.content).strip().lower()
            elif isinstance(result, dict):
                sentiment = result.get("content", result.get("text", "")).strip().lower()
            else:
                sentiment = str(result).strip().lower()
            
            # Normalize response
            if "positive" in sentiment:
                return "positive"
            elif "negative" in sentiment:
                return "negative"
            else:
                return "neutral"
        except (RateLimitError, APIError) as e:
            # Re-raise rate limit errors to trigger retry
            raise
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}", exc_info=False)
            return "neutral"
    
    @retry_with_exponential_backoff(max_retries=5, initial_delay=2.0, max_delay=120.0)
    def extract_topics(self, text: str) -> str:
        """
        Extract topics from text.
        
        Args:
            text: Text to extract topics from
            
        Returns:
            Comma-separated topics
        """
        if not text or len(text.strip()) == 0:
            return ""
        
        try:
            # Use prompt + LLM directly (LangChain v1.x style)
            chain = self.topic_prompt | self.llm
            result = chain.invoke({"text": text})
            
            # Handle AIMessage object (LangChain v1.x response format)
            if hasattr(result, 'content'):
                topics = str(result.content).strip()
            elif isinstance(result, dict):
                topics = result.get("content", result.get("text", "")).strip()
            else:
                topics = str(result).strip()
            return topics
        except (RateLimitError, APIError) as e:
            # Re-raise rate limit errors to trigger retry
            raise
        except Exception as e:
            logger.error(f"Error extracting topics: {e}", exc_info=False)
            return ""
    
    @retry_with_exponential_backoff(max_retries=5, initial_delay=2.0, max_delay=120.0)
    def generate_summary(self, text: str) -> str:
        """
        Generate one-sentence summary.
        
        Args:
            text: Text to summarize
            
        Returns:
            One-sentence summary
        """
        if not text or len(text.strip()) == 0:
            return ""
        
        try:
            # Use prompt + LLM directly (LangChain v1.x style)
            chain = self.summary_prompt | self.llm
            result = chain.invoke({"text": text})
            
            # Handle AIMessage object (LangChain v1.x response format)
            if hasattr(result, 'content'):
                summary = str(result.content).strip()
            elif isinstance(result, dict):
                summary = result.get("content", result.get("text", "")).strip()
            else:
                summary = str(result).strip()
            return summary
        except (RateLimitError, APIError) as e:
            # Re-raise rate limit errors to trigger retry
            raise
        except Exception as e:
            logger.error(f"Error generating summary: {e}", exc_info=False)
            return ""
    
    def enrich_record(self, text: str, delay_between_calls: float = 1.0) -> Dict[str, str]:
        """
        Enrich a single record with all LLM-generated fields.
        
        Args:
            text: Text to enrich
            delay_between_calls: Delay in seconds between each API call (default: 1.0)
            
        Returns:
            Dictionary with sentiment, topics, and summary
        """
        try:
            # Make API calls with delays between them to avoid rate limiting
            # Retry logic is handled by the decorator on each method
            sentiment = self.analyze_sentiment(text)
            time.sleep(delay_between_calls)
            
            topics = self.extract_topics(text)
            time.sleep(delay_between_calls)
            
            summary = self.generate_summary(text)
            
            return {
                'sentiment': sentiment,
                'topics': topics,
                'summary': summary
            }
        except (RateLimitError, APIError) as e:
            # If retries are exhausted, log and return defaults
            logger.error(f"Rate limit error in enrich_record after retries: {e}")
            return {
                'sentiment': 'neutral',
                'topics': '',
                'summary': ''
            }
        except Exception as e:
            logger.error(f"Error in enrich_record: {e}")
            # Return default values on any error
            return {
                'sentiment': 'neutral',
                'topics': '',
                'summary': ''
            }


def enrich_dataframe(df, text_column: str = 'review_text', batch_size: int = 5, 
                     delay_between_calls: float = 0.5, delay_between_records: float = 1.0,
                     max_consecutive_errors: int = 5):
    """
    Enrich a DataFrame with LLM-generated fields.
    
    Args:
        df: DataFrame to enrich
        text_column: Name of the column containing text to enrich
        batch_size: Number of records to process before printing progress
        delay_between_calls: Delay in seconds between each API call within a record (default: 2.0)
        delay_between_records: Delay in seconds between processing records (default: 3.0)
        max_consecutive_errors: Maximum consecutive errors before pausing longer (default: 5)
        
    Returns:
        Enriched DataFrame
    """
    enricher = LLMEnricher()
    
    df = df.copy()
    
    # Initialize new columns
    df['sentiment'] = ""
    df['topics'] = ""
    df['summary'] = ""
    
    # Process each row
    total = len(df)
    consecutive_errors = 0
    adaptive_delay = delay_between_records
    
    for idx, row in df.iterrows():
        text = row.get(text_column, "")
        
        if text and len(text.strip()) > 0:
            try:
                enriched = enricher.enrich_record(text, delay_between_calls=delay_between_calls)
                df.at[idx, 'sentiment'] = enriched.get('sentiment', 'neutral')
                df.at[idx, 'topics'] = enriched.get('topics', '')
                df.at[idx, 'summary'] = enriched.get('summary', '')
                
                # Reset consecutive errors on success
                consecutive_errors = 0
                adaptive_delay = delay_between_records
                
                # Delay between records to avoid rate limiting (3 API calls per record)
                time.sleep(adaptive_delay)
                
            except (RateLimitError, APIError) as e:
                consecutive_errors += 1
                logger.warning(
                    f"Rate limit error at index {idx} (consecutive: {consecutive_errors}). "
                    f"Increasing delay..."
                )
                
                # Exponential backoff for consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    adaptive_delay = min(adaptive_delay * 2, 60.0)  # Cap at 60 seconds
                    logger.warning(f"Increased adaptive delay to {adaptive_delay:.1f} seconds")
                
                # Set default values on error
                df.at[idx, 'sentiment'] = 'neutral'
                df.at[idx, 'topics'] = ''
                df.at[idx, 'summary'] = ''
                
                # Longer delay after rate limit error
                time.sleep(adaptive_delay * 2)
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error enriching record at index {idx}: {e}", exc_info=False)
                # Set default values on error
                df.at[idx, 'sentiment'] = 'neutral'
                df.at[idx, 'topics'] = ''
                df.at[idx, 'summary'] = ''
                time.sleep(adaptive_delay)
        else:
            # Set default values for empty text
            df.at[idx, 'sentiment'] = 'neutral'
            df.at[idx, 'topics'] = ''
            df.at[idx, 'summary'] = ''
        
        # Log progress
        if (idx + 1) % batch_size == 0 or (idx + 1) == total:
            logger.info(
                f"Enriched {idx + 1}/{total} records... "
                f"(Current delay: {adaptive_delay:.1f}s, Consecutive errors: {consecutive_errors})"
            )
    
    logger.info(f"Enrichment complete: {total} records processed")
    return df


if __name__ == "__main__":
    # Test enrichment
    from ingest import load_structured_data, load_unstructured_data
    from clean import clean_structured_data, clean_unstructured_data
    from merge import merge_data
    
    structured = load_structured_data()
    unstructured = load_unstructured_data()
    
    cleaned_structured = clean_structured_data(structured)
    cleaned_unstructured = clean_unstructured_data(unstructured)
    
    merged = merge_data(cleaned_structured, cleaned_unstructured)
    
    # Enrich only records with review text
    records_with_reviews = merged[merged['review_text'] != ''].copy()
    
    if len(records_with_reviews) > 0:
        logger.info(f"\nEnriching {len(records_with_reviews)} records with reviews...")
        enriched = enrich_dataframe(records_with_reviews, text_column='review_text')
        logger.info("\nSample enriched record:")
        logger.info(enriched.head(1).to_dict('records')[0])
    else:
        logger.info("No records with reviews to enrich")

