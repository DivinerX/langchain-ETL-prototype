
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

- **Run the Data Pipeline**: After deployment, you **MUST** run the data ingestion and enrichment pipeline to populate the database. The database file is created automatically, but it starts empty. Run:
  ```bash
  fly ssh console -C "python src/run_pipeline.py"
  ```
  
  **Important**: The pipeline will:
  - Load and clean the data from `data/structured.csv` and `data/unstructured.json`
  - Enrich it with LLM (sentiment, topics, summaries)
  - Store it in the SQLite database at `/app/enriched_data.db`
  
  **Note**: Make sure your `DB_PATH` environment variable matches where you want the database (default is `/app/enriched_data.db`).
  
  **Debug**: If you see 0 records, check the debug endpoint:
  ```bash
  curl https://your-app.fly.dev/debug/database
  ```
  This will show you:
  - Database file location and size
  - Table structure
  - Record count
  - Sample records (if any)
  - Instructions if the database is empty

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
  
  Then mount the volume and set the database path environment variable:
  ```bash
  # Mount the volume (add this to fly.toml under [mounts])
  # Or use: fly volumes attach data
  ```
  
  Set the database path environment variable:
  ```bash
  fly secrets set DB_PATH=/enriched_data.db
  ```
  
  **Note**: If you're using a volume, make sure the volume is mounted. You can check the volume mount in your `fly.toml` file or by running `fly volumes list`.
  
  **Troubleshooting**: If you're getting empty results from the API:
  
  1. **Check if there's a DB_PATH secret that might be overriding fly.toml:**
     ```bash
     fly secrets list
     ```
     **Important**: If you see `DB_PATH` in the secrets list with a Windows path (like `C:/...`), remove it:
     ```bash
     fly secrets unset DB_PATH
     ```
     Secrets take precedence over environment variables in `fly.toml`, so a bad secret will override your correct setting.
  
  2. **Find where your database file actually is:**
     ```bash
     fly ssh console -C "find / -name 'enriched_data.db' 2>/dev/null"
     ```
  
  3. **Check the current working directory and look for the database:**
     ```bash
     fly ssh console -C "pwd && ls -la *.db"
     ```
     (On Fly.io, this should be `/app` based on the Dockerfile)
  
  4. **Set the DB_PATH correctly:**
     
     **Option A: Use fly.toml (recommended for fixed paths)**
     - Edit `fly.toml` and set `DB_PATH = '/app/enriched_data.db'` in the `[env]` section
     - This is already done if you followed the setup
     
     **Option B: Use secrets (if you need different paths per environment)**
     ```bash
     # If database is in /app (default):
     fly secrets set DB_PATH=/app/enriched_data.db
     
     # If database is in root:
     fly secrets set DB_PATH=/enriched_data.db
     
     # If using a volume mounted at /data:
     fly secrets set DB_PATH=/data/enriched_data.db
     ```
     **Note**: Secrets override fly.toml env vars, so if you set a secret, it will be used instead.
  
  5. **Check the health endpoint** (shows database path, environment info, and record count):
     ```bash
     curl https://your-app.fly.dev/health
     ```
     This will show:
     - The exact path being used
     - The DB_PATH environment variable value
     - Whether the file exists
     - The record count
  
  6. **Verify the database has data:**
     ```bash
     # Replace /app/enriched_data.db with the actual path from step 2
     fly ssh console -C "sqlite3 /app/enriched_data.db 'SELECT COUNT(*) FROM enriched_records;'"
     ```
  
  7. **Redeploy after making changes:**
     ```bash
     fly deploy
     ```

- **Free Tier Limits**: 
  - 256MB RAM
  - Shared CPU
  - 3GB storage
  - Auto-stops when idle (auto-starts on request)

- **Health Checks**: The application includes a `/health` endpoint that Fly.io monitors automatically.

- **API Documentation**: Once deployed, access interactive API docs at `https://your-app.fly.dev/docs`

### Deploying Streamlit Frontend (Optional)

The current deployment only includes the FastAPI backend. To also deploy the Streamlit frontend:

**Option 1: Deploy Streamlit as a Separate App** (Recommended for free tier)

1. **Deploy the FastAPI backend first** (using the steps above)

2. **Deploy Streamlit as a separate app:**
   ```bash
   fly launch --config fly.streamlit.toml --dockerfile Dockerfile.streamlit
   ```
   
   When prompted:
   - **App name**: Use a different name like `product-review-streamlit`
   - **Region**: Use the same region as your FastAPI app
   - **Postgres/Redis**: Type `n` for both
   - **Deploy now?**: Type `n`

3. **Set the API URL** (replace with your FastAPI app URL):
   ```bash
   fly secrets set API_BASE_URL=https://product-review-prototype.fly.dev
   ```

4. **Deploy Streamlit:**
   ```bash
   fly deploy --config fly.streamlit.toml --dockerfile Dockerfile.streamlit
   ```

**Option 2: Access API Only**

You can use the FastAPI endpoints directly:
- API: `https://your-app.fly.dev`
- Interactive docs: `https://your-app.fly.dev/docs`
- Health check: `https://your-app.fly.dev/health`

The Streamlit app is designed to work with the API, so you can run it locally and point it to your deployed API, or deploy it separately as shown above.
