# UEE Ingestion & Conversation API

A FastAPI service that:
1. Accepts PDF uploads, extracts semantic text chunks (via Chonkie + PyMuPDF), generates vector embeddings (via HuggingFace Inference API), and stores everything in MongoDB.
2. Provides a **conversation endpoint** for asking RAG-powered questions about Egyptian artifacts grounded in the ingested UEE corpus.

---

## Project Structure

```
uee_ingestion_api/
├── extract_uee_articles_v3.py   # Original extraction script (used as a library)
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app, lifespan, endpoints (/ingest, /conversation, /health)
│   ├── config.py          # Settings (Pydantic BaseSettings, reads .env)
│   ├── models.py          # Pydantic request/response schemas for conversation
│   ├── prompt_builder.py  # ★ Single place for system prompt & embedding query logic
│   ├── chat_client.py     # SBG chat model API integration
│   ├── extractor.py       # Async wrapper around extraction script
│   ├── embedder.py        # HuggingFace embedding calls (async, semaphore-limited)
│   └── database.py        # Motor (async MongoDB) client + $vectorSearch pipeline
├── .env.example           # Template for environment variables
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure secrets & parameters

Copy `.env.example` to `.env` and populate your variables:

```dotenv
HF_TOKEN=hf_...
HF_EMBEDDING_MODEL=microsoft/harrier-oss-v1-0.6b

MONGODB_URI=mongodb+srv://...
MONGODB_DB=uee_db
MONGODB_COLLECTION=chunks
MONGODB_VECTOR_INDEX=vector_index

SBG_API_KEY=your_sbg_api_key
SBG_BASE_URL=https://your-sbg-endpoint/api/v1
LLM_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
LLM_MAX_TOKENS=1024
```

### 3. Start the server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

---

## Endpoints

### `GET /health`

Liveness check.

```json
{ "status": "healthy" }
```

---

### `POST /ingest`

Upload a PDF file for ingestion.

**Request** — `multipart/form-data`:

| Field  | Type | Description         |
|--------|------|---------------------|
| `file` | File | The PDF to ingest   |

**Success response** (`200 OK`):

```json
{
  "status": "ok",
  "file_name": "example.pdf",
  "article_id": "example_article-abc123def456",
  "title": "Block Statue",
  "page_count": 5,
  "chunks_extracted": 42,
  "chunks_stored": 42
}
```

---

### `POST /conversation`

Ask a question about an artifact with RAG retrieval over the UEE database.

**Request** — `application/json`:

```json
{
  "artifact": {
    "name": "Block Statue",
    "period": "Middle Kingdom",
    "material": "Granite",
    "place_of_discovery": "Karnak"
  },
  "messages": [
    { "role": "user", "content": "What are block statues used for?" },
    { "role": "assistant", "content": "Block statues were used as votive offerings..." }
  ],
  "question": "Were they also found in Nubia?"
}
```

**Success response** (`200 OK`):

```json
{
  "answer": "Yes, block statues were also placed in Nubian temples...",
  "retrieved_chunks": 10,
  "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
  "sources": [
    {
      "chunk_id": "eScholarship_UC_item_3f23c0q9-1234:chunk-0011",
      "title": "Block Statue",
      "text": "The statues in these two illustrations...",
      "page_start": 3,
      "page_end": 4,
      "score": 0.892
    }
  ]
}
```

---

## Prompt Customization

All prompt construction is decoupled into [`app/prompt_builder.py`](file:///c:/RuyaGraduation/uee_ingestion_api/app/prompt_builder.py):

1. **`build_query_text(artifact, question)`**: Constructs the string embedded to search MongoDB Atlas Vector Search.
2. **`build_system_prompt(artifact, question, chunks)`**: Constructs the system prompt passed to the LLM (includes role, artifact info, retrieved chunks, and answering instructions).

---

## Interactive API Docs

Once running:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
