
# AI-Powered Data Enrichment & Semantic Search Pipeline

## Overview

This repository contains a **production-inspired AI workflow prototype** demonstrating how to ingest, enrich, store, and expose structured and unstructured data using modern LLM-centric architecture.

The project reflects real-world AI system design principles while remaining intentionally lightweight and easy to reason about for a technical assessment.

---

## Objectives

This prototype satisfies the following requirements:

- Ingest data from **two distinct sources**
  - One structured data source
  - One unstructured text-based source
- Clean, normalize, and combine both data sources
- Enrich data using a Large Language Model
- Orchestrate the workflow using Python and LangChain
- Persist enriched data in a relational database
- Enable semantic search using a vector database
- Expose results through an API and a basic user interface

---

## Technology Stack

### Backend
- **Language:** Python 3.10+
- **API Framework:** FastAPI
- **LLM Provider:** OpenAI
- **LLM Orchestration:** LangChain
- **Relational Database:** SQLite
- **Vector Database:** Pinecone
- **Data Processing:** Pandas
- **Text Matching:** RapidFuzz

### Frontend
- **Framework:** Streamlit

---

## Data Sources

### Structured Source
A CSV-based dataset representing structured entities such as products, tickets, or records.

Typical fields:
- ID
- Name
- Category
- Price or metadata

Loaded and processed using Pandas.

### Unstructured Source
A text-based dataset such as:
- User reviews
- Support tickets
- Free-form notes or comments

Stored as JSON or text files and cleaned prior to enrichment.

This design explicitly demonstrates handling heterogeneous data sources.

---

## Workflow Architecture

### 1. Ingestion
- Load structured CSV data
- Load unstructured text documents

### 2. Cleaning and Normalization
- Normalize text and categorical values
- Remove duplicates and invalid rows
- Apply fuzzy matching where required to associate documents with structured entities

### 3. LLM Enrichment
OpenAI models are used via LangChain to perform:

- Sentiment classification
- Topic extraction
- One-sentence summarization
- Metadata normalization

LangChain chains provide deterministic execution and extensibility.

---

## Storage Layer

### Relational Storage
- **SQLite**

Stores the final enriched dataset, including:
- Structured metadata
- Source text
- LLM-generated sentiment
- Extracted topics
- Generated summaries

### Vector Storage
- **Pinecone**

Used to:
- Store embeddings for unstructured text
- Enable semantic similarity search
- Support hybrid search workflows when combined with relational filters

Embeddings are generated during ingestion and upserted with metadata.

---

## API Layer

The backend exposes a RESTful API using FastAPI.

### Available Endpoints

- `GET /records`
- `GET /records/{id}`
- `GET /search?query=`

Interactive API documentation is available at `/docs`.

---

## Frontend

### Streamlit Dashboard

A lightweight Streamlit interface provides:
- Browsing of enriched records
- Filtering by sentiment and topic
- Display of LLM-generated summaries
- Semantic search powered by Pinecone

The frontend consumes data exclusively through the FastAPI backend.

---

## Project Structure

```
ai-workflow-demo/
│
├── data/
│   ├── structured.csv
│   └── unstructured.json
│
├── src/
│   ├── ingest.py
│   ├── clean.py
│   ├── merge.py
│   ├── enrich_llm.py
│   ├── embeddings.py
│   ├── database.py
│   ├── api.py
│   └── run_pipeline.py
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .env.example
```

---

## Setup & Execution

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file:
```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
```

### 3. Run the Pipeline
```
python src/run_pipeline.py
```

### 4. Start the API
```
uvicorn src.api:app --reload
```

### 5. Launch the Frontend
```
streamlit run streamlit_app.py
```

---

## Scaling Considerations

If extended to production, this system could scale by:

- Migrating SQLite to PostgreSQL or a data warehouse
- Running LLM enrichment as async batch jobs
- Introducing a queue-based ingestion pipeline
- Caching LLM outputs to control cost
- Expanding Pinecone usage for RAG workflows
- Adding observability and cost monitoring
- Deploying via CI/CD with containerized services

---

## Summary

This project demonstrates:

- Multi-source data ingestion
- Structured and unstructured data integration
- LLM-powered enrichment using OpenAI
- Workflow orchestration with LangChain
- Separation of relational and vector storage
- Clean API design
- Lightweight and usable frontend

It reflects production-minded AI engineering principles in a concise, extensible prototype.
