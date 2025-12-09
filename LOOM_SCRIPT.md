# Loom Script: AI-Powered Data Enrichment Pipeline
## Principal AI Engineer Presentation (7-10 minutes)

---

## [0:00-0:30] Introduction & Project Overview

**[Show: Project directory structure in IDE]**

"Hi, I'm [Your Name], Principal AI Engineer. Today I'll walk you through an AI-powered data enrichment and semantic search pipeline that demonstrates production-inspired architecture for handling heterogeneous data sources with LLM integration.

This project solves a common enterprise challenge: ingesting structured and unstructured data from multiple sources, enriching it with AI-generated insights, and making it searchable through both traditional and semantic search capabilities.

**[Show: README.md]**

The system processes product data from CSV files and customer reviews from JSON, then uses OpenAI's GPT models via LangChain to extract sentiment, topics, and summaries. The enriched data is stored in both SQLite for relational queries and Pinecone for vector similarity search, exposed through a FastAPI backend with a Streamlit frontend."

---

## [0:30-1:30] Architecture & Design Philosophy

**[Show: Project structure in file explorer]**

"Let me walk you through the architecture. The system follows a modular, pipeline-based design with clear separation of concerns.

**[Show: src/ directory]**

We have seven core modules:
- **Ingestion**: Loads structured CSV and unstructured JSON data
- **Cleaning**: Normalizes and validates data
- **Merging**: Associates unstructured reviews with structured products using fuzzy matching
- **LLM Enrichment**: Uses LangChain to orchestrate OpenAI API calls for sentiment, topics, and summaries
- **Embeddings**: Generates vector embeddings for semantic search
- **Database**: Manages SQLite persistence
- **API**: FastAPI REST endpoints

**[Show: run_pipeline.py - main orchestrator]**

The pipeline orchestrator, `run_pipeline.py`, coordinates all seven steps sequentially. This design makes it easy to reason about, test, and extend. Each module can be run independently or as part of the full pipeline."

---

## [1:30-2:30] Technology Stack & Design Decisions

**[Show: requirements.txt]**

"Our technology stack reflects production-minded choices:

- **FastAPI** for the API layer - modern, async-capable, with automatic OpenAPI docs
- **LangChain** for LLM orchestration - provides retry logic, prompt templating, and extensibility
- **OpenAI GPT-4o-mini** - cost-effective for batch processing while maintaining quality
- **SQLite** for relational storage - lightweight but production-ready schema
- **Pinecone** for vector search - managed service that scales
- **Streamlit** for rapid frontend prototyping

**[Show: src/enrich_llm.py - retry logic]**

One key design decision: robust error handling. Notice the exponential backoff decorator for rate limit handling. We make three API calls per record - sentiment, topics, and summary - so we need intelligent retry logic with jitter to avoid overwhelming the API.

**[Show: retry_with_exponential_backoff function]**

This decorator handles 429 errors gracefully, with configurable delays and maximum retries. In production, we'd add circuit breakers and caching to reduce API costs."

---

## [2:30-3:45] Data Flow Deep Dive

**[Show: data/structured.csv and data/unstructured.json]**

"Let's trace the data flow. We start with two sources:

**[Show: structured.csv]**

Structured data: product IDs, names, categories, and prices in CSV format.

**[Show: unstructured.json]**

Unstructured data: free-form customer reviews in JSON.

**[Show: src/merge.py or code showing merge logic]**

The merge step uses fuzzy matching - specifically RapidFuzz - to associate reviews with products when IDs don't perfectly align. This handles real-world data quality issues.

**[Show: src/enrich_llm.py - LLMEnricher class]**

For enrichment, we use LangChain's prompt templates and chain composition. Each enrichment task - sentiment, topics, summary - has a dedicated prompt template. The chain pattern makes it easy to swap models or add new enrichment tasks.

**[Show: _setup_chains method]**

Notice the prompts are concise and structured to get deterministic outputs. We use temperature 0.0 for consistency."

---

## [3:45-5:00] Storage Layer & Dual Database Strategy

**[Show: src/database.py]**

"Now, the storage layer. We use a dual-database strategy:

**[Show: SQLite table schema]**

SQLite stores the complete enriched records with all metadata - product info, review text, sentiment, topics, summaries. This supports traditional SQL queries, filtering, and joins.

**[Show: src/embeddings.py]**

For semantic search, we generate embeddings using OpenAI's text-embedding-3-small model. These are 512-dimensional vectors that capture semantic meaning.

**[Show: run_pipeline.py - Pinecone upsert section]**

We upsert these embeddings to Pinecone with metadata. This enables semantic similarity search - finding reviews that are conceptually similar even if they don't share exact keywords.

**[Show: upsert_to_pinecone function]**

The upsert logic handles dimension mismatches gracefully - if the index expects a different dimension, we regenerate embeddings. This is important for production systems where index configurations might change."

---

## [5:00-6:15] API Layer & Frontend

**[Show: src/api.py]**

"The API layer exposes three main endpoints:

**[Show: API endpoints]**

- `GET /records` - Retrieve all enriched records
- `GET /records/{id}` - Get a specific record
- `GET /search` - Search with query, sentiment, or topic filters

**[Show: CORS middleware]**

We've enabled CORS for the Streamlit frontend. In production, we'd restrict this to specific origins.

**[Show: FastAPI docs endpoint - /docs]**

FastAPI automatically generates interactive API documentation at `/docs` - this is invaluable for integration and testing.

**[Show: streamlit_app.py if available, or describe]**

The Streamlit frontend provides a lightweight UI for browsing records, filtering by sentiment and topics, and performing semantic searches. It consumes data exclusively through the API, demonstrating proper separation of concerns."

---

## [6:15-7:30] Production Considerations & Scaling

**[Show: README.md - Scaling Considerations section]**

"Let's discuss production readiness. This prototype demonstrates core patterns, but scaling would require:

**[Show: scaling considerations]**

1. **Database Migration**: SQLite to PostgreSQL or a data warehouse for concurrent access and larger datasets

2. **Async Processing**: LLM enrichment should run as async batch jobs, not blocking the main pipeline. We'd use Celery or similar.

3. **Queue-Based Ingestion**: For high-volume data, we'd introduce Kafka or RabbitMQ to decouple ingestion from processing.

4. **Caching**: LLM outputs should be cached to control costs. We'd use Redis to cache embeddings and enrichment results.

5. **Observability**: Add logging, metrics, and tracing - especially for API costs and latency.

6. **Error Handling**: More sophisticated error handling with dead-letter queues for failed enrichments.

7. **Containerization**: Dockerize services for consistent deployments.

The current architecture makes these extensions straightforward - each module is already well-separated."

---

## [7:30-8:30] Key Features & Highlights

**[Show: Code examples of key features]**

"Let me highlight a few features that demonstrate production-minded engineering:

**[Show: enrich_llm.py - adaptive delay logic]**

First, adaptive rate limiting. The system monitors consecutive errors and increases delays dynamically. This prevents cascading failures.

**[Show: database.py - duplicate detection]**

Second, duplicate detection. Before inserting records, we check for existing product-review combinations. This prevents data corruption on re-runs.

**[Show: run_pipeline.py - dimension handling]**

Third, dimension validation for embeddings. The system detects dimension mismatches between embeddings and Pinecone indexes, automatically regenerating if needed.

**[Show: API response models]**

Fourth, type safety with Pydantic models. All API responses are validated, providing clear error messages and API documentation."

---

## [8:30-9:30] Demo Walkthrough (Optional)

**[If doing live demo, show terminal/IDE]**

"Let me quickly demonstrate the pipeline in action:

**[Run: python src/run_pipeline.py]**

When we run the pipeline, you'll see progress logs for each step. The LLM enrichment step shows progress every 5 records, including current delay settings and error counts.

**[Show: API running]**

Once complete, we start the API server. The `/docs` endpoint gives us interactive API documentation.

**[Show: API calls]**

We can query all records, filter by sentiment, or search by topic. The semantic search would query Pinecone for similar reviews.

**[Show: Streamlit app if available]**

The Streamlit frontend provides a user-friendly interface for exploring the enriched data."

---

## [9:30-10:00] Conclusion & Next Steps

**[Show: Project overview]**

"In summary, this project demonstrates:

- **Multi-source data ingestion** with fuzzy matching
- **LLM-powered enrichment** using LangChain orchestration
- **Dual storage strategy** - relational and vector databases
- **Production-ready patterns** - error handling, retries, type safety
- **Clean API design** with automatic documentation
- **Extensible architecture** that can scale to production workloads

The codebase is intentionally lightweight and easy to reason about, making it perfect for technical assessments while still reflecting real-world AI system design principles.

For next steps, I'd recommend:
1. Adding comprehensive unit and integration tests
2. Implementing the caching layer for cost control
3. Adding monitoring and alerting
4. Containerizing with Docker
5. Setting up CI/CD pipelines

Thank you for your time. I'm happy to answer any questions about the architecture, design decisions, or implementation details."

---

## Presentation Tips

1. **Pacing**: Aim for ~1 minute per major section. Adjust demo time based on available time.
2. **Screen Focus**: Keep code visible but zoomed appropriately. Use IDE's file explorer to navigate.
3. **Code Highlighting**: Use your IDE's highlighting to point out specific functions or patterns.
4. **Transitions**: Use phrases like "Let's look at...", "Notice here...", "The key point is..."
5. **Technical Depth**: Balance high-level architecture with specific implementation details.
6. **Energy**: Maintain enthusiasm - this is a strong technical demonstration!

---

## Key Talking Points to Emphasize

- **Modularity**: Each component is independently testable and replaceable
- **Error Resilience**: Production-grade error handling and retry logic
- **Cost Awareness**: Rate limiting and potential caching strategies
- **Extensibility**: Easy to add new enrichment tasks or data sources
- **Type Safety**: Pydantic models ensure API contract compliance
- **Observability**: Logging throughout for debugging and monitoring
- **Separation of Concerns**: Clear boundaries between ingestion, processing, storage, and API layers

