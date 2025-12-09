"""
LLM enrichment module using OpenAI and LangChain for sentiment analysis, topic extraction, and summarization.
"""
import os
import logging
from typing import Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LLMEnricher:
    """Class for enriching data using LLM via LangChain."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
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
        """Setup LangChain chains for different enrichment tasks."""
        
        # Sentiment analysis chain
        sentiment_prompt = ChatPromptTemplate.from_template(
            "Analyze the sentiment of the following text and respond with ONLY one word: "
            "positive, negative, or neutral.\n\nText: {text}\n\nSentiment:"
        )
        self.sentiment_chain = LLMChain(llm=self.llm, prompt=sentiment_prompt)
        
        # Topic extraction chain
        topic_prompt = ChatPromptTemplate.from_template(
            "Extract the main topics or themes from the following text. "
            "Respond with 2-3 comma-separated topics. Be concise.\n\nText: {text}\n\nTopics:"
        )
        self.topic_chain = LLMChain(llm=self.llm, prompt=topic_prompt)
        
        # Summarization chain
        summary_prompt = ChatPromptTemplate.from_template(
            "Provide a one-sentence summary of the following text. "
            "Be concise and informative.\n\nText: {text}\n\nSummary:"
        )
        self.summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)
    
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
            # Try invoke (newer LangChain) first, fallback to run (older LangChain)
            try:
                result = self.sentiment_chain.invoke({"text": text})
            except AttributeError:
                result = self.sentiment_chain.run(text=text)
            
            # Handle both dict and string responses
            if isinstance(result, dict):
                sentiment = result.get("text", "").strip().lower()
            else:
                sentiment = str(result).strip().lower()
            
            # Normalize response
            if "positive" in sentiment:
                return "positive"
            elif "negative" in sentiment:
                return "negative"
            else:
                return "neutral"
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return "neutral"
    
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
            # Try invoke (newer LangChain) first, fallback to run (older LangChain)
            try:
                result = self.topic_chain.invoke({"text": text})
            except AttributeError:
                result = self.topic_chain.run(text=text)
            
            # Handle both dict and string responses
            if isinstance(result, dict):
                topics = result.get("text", "").strip()
            else:
                topics = str(result).strip()
            return topics
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return ""
    
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
            # Try invoke (newer LangChain) first, fallback to run (older LangChain)
            try:
                result = self.summary_chain.invoke({"text": text})
            except AttributeError:
                result = self.summary_chain.run(text=text)
            
            # Handle both dict and string responses
            if isinstance(result, dict):
                summary = result.get("text", "").strip()
            else:
                summary = str(result).strip()
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return ""
    
    def enrich_record(self, text: str) -> Dict[str, str]:
        """
        Enrich a single record with all LLM-generated fields.
        
        Args:
            text: Text to enrich
            
        Returns:
            Dictionary with sentiment, topics, and summary
        """
        return {
            'sentiment': self.analyze_sentiment(text),
            'topics': self.extract_topics(text),
            'summary': self.generate_summary(text)
        }


def enrich_dataframe(df, text_column: str = 'review_text', batch_size: int = 5):
    """
    Enrich a DataFrame with LLM-generated fields.
    
    Args:
        df: DataFrame to enrich
        text_column: Name of the column containing text to enrich
        batch_size: Number of records to process before printing progress
        
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
    for idx, row in df.iterrows():
        text = row.get(text_column, "")
        
        if text and len(text.strip()) > 0:
            enriched = enricher.enrich_record(text)
            df.at[idx, 'sentiment'] = enriched['sentiment']
            df.at[idx, 'topics'] = enriched['topics']
            df.at[idx, 'summary'] = enriched['summary']
        
        # Log progress
        if (idx + 1) % batch_size == 0 or (idx + 1) == total:
            logger.info(f"Enriched {idx + 1}/{total} records...")
    
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

