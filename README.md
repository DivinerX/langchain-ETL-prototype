
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

---

## Deployment to Fly.io

This project includes configuration files for deploying to Fly.io's free tier.

### Prerequisites

1. **Install Fly CLI**
   - **Windows (PowerShell):**
     ```powershell
     powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
     ```
   - **macOS/Linux:**
     ```bash
     curl -L https://fly.io/install.sh | sh
     ```

2. **Login to Fly.io**
   ```bash
   fly auth login
   ```
   This will open a browser window for authentication. Complete the login process.

### Deployment Steps

1. **Create and Launch the Application**
   ```bash
   fly launch
   ```
   
   During the launch process, you'll be prompted with several questions:
   
   - **App name**: Press Enter to use `product-review` (or type a different name)
   - **Region**: Press Enter to use `dfw` (or select a different region)
   - **Postgres database**: Type `n` and press Enter (we're using SQLite)
   - **Redis**: Type `n` and press Enter (not needed for this project)
   - **Deploy now?**: Type `n` and press Enter (we'll set secrets first)
   
   This command creates the app on Fly.io and prepares it for deployment.

2. **Set Environment Variables (Secrets)**
   
   Set your API keys as secrets (these are encrypted and stored securely):
   ```bash
   fly secrets set OPENAI_API_KEY=your_openai_key
   fly secrets set PINECONE_API_KEY=your_pinecone_key
   fly secrets set PINECONE_INDEX_NAME=your_index_name
   ```
   
   Replace `your_openai_key`, `your_pinecone_key`, and `your_index_name` with your actual values.
   
   **Note**: You can set all secrets at once:
   ```bash
   fly secrets set OPENAI_API_KEY=your_openai_key PINECONE_API_KEY=your_pinecone_key PINECONE_INDEX_NAME=your_index_name
   ```

3. **Deploy the Application**
   ```bash
   fly deploy
   ```
   
   This command will:
   - Build the Docker image using the `Dockerfile`
   - Push the image to Fly.io
   - Deploy the application to the cloud
   - Start the application
   
   The deployment process may take a few minutes. You'll see build logs and deployment progress.

4. **Verify Deployment**
   ```bash
   fly status
   fly logs
   ```

5. **Open Your Application**
   ```bash
   fly open
   ```
   Or visit: `https://product-review.fly.dev`

### Post-Deployment

- **Run the Data Pipeline**: After deployment, you may need to run the data ingestion and enrichment pipeline. You can do this by SSH'ing into the machine:
  ```bash
  fly ssh console
  python src/run_pipeline.py
  ```

- **View Logs**: Monitor your application logs:
  ```bash
  fly logs
  ```

- **Scale Resources** (if needed): The default configuration uses 256MB RAM (free tier). To scale:
  ```bash
  fly scale memory 512
  ```

### Important Notes

- **SQLite Persistence**: The SQLite database is ephemeral by default. For persistent storage, create a volume:
  ```bash
  fly volumes create data --size 1
  ```
  Then update the database path in your code to use the volume mount.

- **Free Tier Limits**: 
  - 256MB RAM
  - Shared CPU
  - 3GB storage
  - Auto-stops when idle (auto-starts on request)

- **Health Checks**: The application includes a `/health` endpoint that Fly.io monitors automatically.

- **API Documentation**: Once deployed, access interactive API docs at `https://your-app.fly.dev/docs`
