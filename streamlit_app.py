"""
Streamlit frontend for browsing and searching enriched data.
"""
import streamlit as st
import pandas as pd
import requests
from typing import List, Dict, Optional

# API base URL
API_BASE_URL = "http://localhost:8000"


def fetch_all_records() -> List[Dict]:
    """Fetch all records from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/records")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching records: {e}")
        return []


def fetch_record_by_id(record_id: int) -> Optional[Dict]:
    """Fetch a specific record by ID."""
    try:
        response = requests.get(f"{API_BASE_URL}/records/{record_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching record: {e}")
        return None


def search_records(query: Optional[str] = None, sentiment: Optional[str] = None, topic: Optional[str] = None) -> List[Dict]:
    """Search records using the API."""
    try:
        params = {}
        if query:
            params['query'] = query
        if sentiment:
            params['sentiment'] = sentiment
        if topic:
            params['topic'] = topic
        
        response = requests.get(f"{API_BASE_URL}/search", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error searching records: {e}")
        return []


# Page configuration
st.set_page_config(
    page_title="AI-Powered Data Enrichment Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 AI-Powered Data Enrichment Dashboard")
st.markdown("Browse and search enriched product reviews with LLM-generated insights")

# Sidebar
st.sidebar.header("Filters")

# Sentiment filter
sentiment_filter = st.sidebar.selectbox(
    "Filter by Sentiment",
    ["All", "positive", "negative", "neutral"]
)

# Topic filter
topic_filter = st.sidebar.text_input("Filter by Topic (partial match)")

# Search query
search_query = st.sidebar.text_input("Search Query")

# Main content
tab1, tab2, tab3 = st.tabs(["📊 All Records", "🔍 Search", "📝 Record Details"])

with tab1:
    st.header("All Enriched Records")
    
    # Fetch records
    if sentiment_filter != "All" or topic_filter or search_query:
        records = search_records(
            query=search_query if search_query else None,
            sentiment=sentiment_filter if sentiment_filter != "All" else None,
            topic=topic_filter if topic_filter else None
        )
    else:
        records = fetch_all_records()
    
    if records:
        df = pd.DataFrame(records)
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("With Reviews", len(df[df['review_text'].notna() & (df['review_text'] != '')]))
        with col3:
            if 'sentiment' in df.columns:
                positive_count = len(df[df['sentiment'] == 'positive'])
                st.metric("Positive", positive_count)
        with col4:
            if 'sentiment' in df.columns:
                negative_count = len(df[df['sentiment'] == 'negative'])
                st.metric("Negative", negative_count)
        
        # Display records table
        st.subheader("Records")
        
        # Select columns to display
        display_columns = ['id', 'product_name', 'category', 'price', 'sentiment', 'topics']
        available_columns = [col for col in display_columns if col in df.columns]
        
        if available_columns:
            st.dataframe(
                df[available_columns],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Display detailed view for selected record
        if len(df) > 0:
            st.subheader("Record Details")
            selected_id = st.selectbox(
                "Select a record to view details",
                options=df['id'].tolist() if 'id' in df.columns else []
            )
            
            if selected_id:
                record = fetch_record_by_id(selected_id)
                if record:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Product Information**")
                        st.write(f"**ID:** {record.get('product_id', 'N/A')}")
                        st.write(f"**Name:** {record.get('product_name', 'N/A')}")
                        st.write(f"**Category:** {record.get('category', 'N/A')}")
                        st.write(f"**Price:** ${record.get('price', 'N/A')}")
                    
                    with col2:
                        st.write("**Review Analysis**")
                        st.write(f"**Sentiment:** {record.get('sentiment', 'N/A')}")
                        st.write(f"**Topics:** {record.get('topics', 'N/A')}")
                        st.write(f"**Summary:** {record.get('summary', 'N/A')}")
                    
                    st.write("**Review Text**")
                    st.write(record.get('review_text', 'No review text available'))
    else:
        st.info("No records found. Make sure the API is running and the pipeline has been executed.")

with tab2:
    st.header("Semantic Search")
    
    search_input = st.text_input("Enter search query", key="search_input")
    
    if search_input:
        results = search_records(query=search_input)
        
        if results:
            st.success(f"Found {len(results)} matching records")
            
            for record in results:
                with st.expander(f"Record {record.get('id')}: {record.get('product_name', 'N/A')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Category:** {record.get('category', 'N/A')}")
                        st.write(f"**Price:** ${record.get('price', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Sentiment:** {record.get('sentiment', 'N/A')}")
                        st.write(f"**Topics:** {record.get('topics', 'N/A')}")
                    
                    st.write("**Summary:**")
                    st.write(record.get('summary', 'N/A'))
                    
                    st.write("**Review Text:**")
                    st.write(record.get('review_text', 'N/A'))
        else:
            st.warning("No results found")
    else:
        st.info("Enter a search query to find relevant records")

with tab3:
    st.header("Record Details by ID")
    
    record_id = st.number_input("Enter Record ID", min_value=1, value=1, step=1)
    
    if st.button("Fetch Record"):
        record = fetch_record_by_id(record_id)
        
        if record:
            st.success("Record found!")
            
            # Display in a nice format
            st.subheader("Product Information")
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.write(f"**Product ID:** {record.get('product_id', 'N/A')}")
                st.write(f"**Product Name:** {record.get('product_name', 'N/A')}")
            
            with info_col2:
                st.write(f"**Category:** {record.get('category', 'N/A')}")
                st.write(f"**Price:** ${record.get('price', 'N/A')}")
            
            st.subheader("LLM Enrichment")
            enrichment_col1, enrichment_col2 = st.columns(2)
            
            with enrichment_col1:
                sentiment = record.get('sentiment', 'N/A')
                sentiment_color = {
                    'positive': '🟢',
                    'negative': '🔴',
                    'neutral': '🟡'
                }.get(sentiment, '⚪')
                st.write(f"**Sentiment:** {sentiment_color} {sentiment}")
                st.write(f"**Topics:** {record.get('topics', 'N/A')}")
            
            with enrichment_col2:
                st.write(f"**Summary:**")
                st.write(record.get('summary', 'N/A'))
            
            st.subheader("Review Text")
            st.write(record.get('review_text', 'No review text available'))
        else:
            st.error(f"Record with ID {record_id} not found")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**API Status**")
try:
    response = requests.get(f"{API_BASE_URL}/health")
    if response.status_code == 200:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Error")
except:
    st.sidebar.error("❌ API Not Available")
    st.sidebar.info("Make sure the API is running:\n`uvicorn src.api:app --reload`")

