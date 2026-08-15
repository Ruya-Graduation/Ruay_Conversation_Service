# UEE Ingestion & Conversation API

A FastAPI service that:
1. Accepts PDF uploads, extracts semantic text chunks (via Chonkie + PyMuPDF), generates vector embeddings (via HuggingFace Inference API), and stores everything in MongoDB.
2. Provides a **conversation endpoint** for asking RAG-powered questions about Egyptian artifacts grounded in the ingested UEE corpus.
3. **NEW:** Provides **artifact detection endpoints** using YOLO for image-based recognition of 84 Egyptian museum artifacts.

---

## Project Structure

```
uee_ingestion_api/
├── extract_uee_articles_v3.py   # Original extraction script (used as a library)
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app, lifespan, endpoints (/ingest, /conversation, /detect-artifact, /health)
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

**Note:** The new artifact detection feature requires YOLO and OpenCV dependencies. These have been added to `requirements.txt`:
- `ultralytics>=8.0.220`
- `opencv-python>=4.8.1.78`
- `Pillow>=10.1.0`
- `numpy>=1.24.3`

### 2. Place YOLO Model

The YOLO artifact detection model (`best.pt`) is already included in the `app/` directory. This is a trained model (49.71 MB) that can detect 84 different Egyptian museum artifacts.

If you need to update or replace the model, place your `best.pt` file in:
```
c:\RuyaGraduation\python\uee_ingestion_api\app\best.pt
```

### 3. Configure secrets & parameters

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

### 4. Start the server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

---

## Endpoints

### `GET /health`

Liveness check.

```json
{ 
  "status": "healthy",
  "yolo_model_loaded": true
}
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

## 🆕 Artifact Detection Endpoints (YOLO)

### `POST /detect-artifact`

Detect Egyptian artifacts in uploaded images using YOLO. Returns the artifact with the highest confidence.

**Request** — `multipart/form-data`:

| Field  | Type        | Description            |
|--------|-------------|------------------------|
| `file` | Image File  | JPG, PNG, etc.         |

**Success response** (`200 OK`):

```json
{
  "artifact_id": "Statue of Tutankhamun",
  "confidence": 0.89,
  "class_id": 80,
  "detections_count": 1
}
```

**No detection response** (`200 OK`):

```json
{
  "artifact_id": null,
  "confidence": 0.0,
  "message": "No artifact detected in image. Try a clearer image or different angle."
}
```

---

### `POST /detect-artifact-detailed`

Detect artifacts with detailed information including all detections and bounding boxes.

**Request** — `multipart/form-data`:

| Field  | Type        | Description            |
|--------|-------------|------------------------|
| `file` | Image File  | JPG, PNG, etc.         |

**Success response** (`200 OK`):

```json
{
  "detections": [
    {
      "artifact_id": "Statue of Tutankhamun",
      "confidence": 0.89,
      "class_id": 80,
      "bbox": {
        "x1": 120.5,
        "y1": 45.2,
        "x2": 450.8,
        "y2": 670.3
      }
    }
  ],
  "count": 1,
  "image_shape": {
    "height": 800,
    "width": 600,
    "channels": 3
  }
}
```

---

### `GET /model-info`

Get information about the loaded YOLO model and available artifacts.

**Success response** (`200 OK`):

```json
{
  "loaded": true,
  "model_path": "c:\\RuyaGraduation\\python\\ImageRecognition\\Egyptian-Museum-Artifact-Detection-main\\backend\\best.pt",
  "num_classes": 84,
  "confidence_threshold": 0.5,
  "model_type": "YOLO",
  "sample_artifacts": [
    "Akhenaten",
    "Amenhotep III",
    "Amenhotep III and Tiye",
    ...
  ],
  "total_artifacts": 84
}
```

---

### `GET /artifacts`

Get complete list of all 84 detectable artifacts.

**Success response** (`200 OK`):

```json
{
  "artifacts": [
    { "class_id": 0, "artifact_name": "Akhenaten" },
    { "class_id": 1, "artifact_name": "Amenhotep III" },
    ...
  ],
  "total": 84
}
```

---

### `GET /artifacts/{class_id}`

Get information about a specific artifact by class ID.

**Example:** `GET /artifacts/80`

**Success response** (`200 OK`):

```json
{
  "class_id": 80,
  "artifact_name": "Statue of Tutankhamun"
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
