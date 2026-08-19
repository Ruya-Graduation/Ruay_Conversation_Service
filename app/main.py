"""
app/main.py
───────────
FastAPI application entry-point.

Startup (lifespan):
  1. Validate required config values.
  2. Initialize Chonkie SemanticChunker (heavy, shared across requests).
  3. Initialize HuggingFace InferenceClient.
  4. Connect to MongoDB and ensure indexes.

Endpoints:
  POST /ingest        – Accept a PDF upload, extract → embed → store.
  POST /conversation  – Artifact + history + question → RAG → LLM answer.
  GET  /health        – Simple liveness check.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
# from fastapi import File, UploadFile
# from fastapi.responses import JSONResponse
from huggingface_hub import InferenceClient
# from ultralytics import YOLO
# import cv2
# import numpy as np

# # Make repo root importable (extract_uee_articles_v3 lives there).
# _ROOT = Path(__file__).resolve().parent.parent
# if str(_ROOT) not in sys.path:
#     sys.path.insert(0, str(_ROOT))

# from extract_uee_articles_v3 import create_chonkie_semantic_chunker  # noqa: E402

from app.config import settings
from app.database import (
    # check_article_exists,
    create_motor_client,
    # ensure_indexes,
    # insert_chunks,
    vector_search,
)
from app.embedder import embed_chunks
# from app.extractor import extract_chunks
from app.chat_client import call_chat, extract_answer
from app.prompt_builder import build_query_text, build_system_prompt, build_user_message
from app.models import (
    ConversationRequest,
    ConversationResponse,
    RetrievedChunk,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Application state (shared across requests) ────────────────────────────────

class AppState:
    # chunker: Any = None
    hf_client: InferenceClient | None = None
    mongo_client: Any = None
    collection: Any = None
    # yolo_model: Any = None  # YOLO model for artifact detection


app_state = AppState()


# # ── YOLO Artifact Detection Configuration ─────────────────────────────────────
#
# # Path to YOLO model (in app directory)
# YOLO_MODEL_PATH = Path(__file__).parent / "best.pt"
#
# # Egyptian Artifact ID mapping - 84 classes
# ARTIFACT_MAPPING = {
#     0: "Akhenaten",
#     1: "Amenhotep III",
#     2: "Amenhotep III and Tiye",
#     3: "Amenhotep III with Plate",
#     4: "Augustus",
#     5: "Bent Pyramid of King Sneferu",
#     6: "Black Granite Bust of Mentuemhat",
#     7: "Bust of Isis",
#     8: "Clossal Head of the god Serapis",
#     9: "Clossal head of Senwosret 1",
#     10: "Coffin of Ahmose I",
#     11: "Colossal Statue of Amenhotep III",
#     12: "Colossal Statue of God Ptah",
#     13: "Colossal Statue of Hormoheb",
#     14: "Colossal Statue of King Senwosret IlI",
#     15: "Colossal Statue of Middle Kingdom King",
#     16: "Colossal Statue of Queen Hatshepsut",
#     17: "Colossal Statue of Ramesses II",
#     18: "Colossal Statue of Ramesses II beloved of Ptah",
#     19: "Colossoi of Memnon",
#     20: "Colossus of Senuseret I",
#     21: "Column of Merenptah",
#     22: "Granite Statue of Osiris",
#     23: "Granite Statue of Tutankhamun",
#     24: "Great Pyramids of Giza",
#     25: "Grey Granite of Ramesses II",
#     26: "Hathor Capital",
#     27: "Hatshepsut face",
#     28: "Head Statue of Amenhotep III",
#     29: "Head Statue of Amenhotep iii",
#     30: "Head of Userkaf",
#     31: "Hor I",
#     32: "Isis with her child",
#     33: "King Amenemhat 3",
#     34: "King Thutmose III",
#     35: "Mask of Thuya",
#     36: "Mask of Tutankhamun",
#     37: "Mask of Yuya",
#     38: "Menkaure Statue",
#     39: "Mentuhotep Nebhetpre",
#     40: "Naos of Senwosert I",
#     41: "Nefertiti",
#     42: "Obelsik Tip of Hatshepsut",
#     43: "Offering table of Amenemhat 6",
#     44: "Pyramid of Djoser",
#     45: "Rhetorical Stela of King Ramesses ll",
#     46: "Seated Statue of Amenhotep III",
#     47: "Seated Statue of Djoser",
#     48: "Seated Statue of God Sekhmet",
#     49: "Seated Statue of Ramesses II",
#     50: "Seated Statue of Ramesses II and God Ptah",
#     51: "Seated Statue of Thutmose III",
#     52: "Senwosret III",
#     53: "Sphinx",
#     54: "Sphinx of Amenmhat III",
#     55: "Sphinx of Kings Ramesses ll - Merenptah",
#     56: "Standing Statue of King Ramses II",
#     57: "Standing Statue of Thutmose III",
#     58: "Statue Head of Akhenaten",
#     59: "Statue of Amenhotep III and God Re-Horakhty",
#     60: "Statue of Amenmhat I",
#     61: "Statue of Amun and King",
#     62: "Statue of Ankhesenamun",
#     63: "Statue of Carcala",
#     64: "Statue of God Ptah Ramesses ll Goddess Sekhmet",
#     65: "Statue of God Ra-Horakhty",
#     66: "Statue of Khafre",
#     67: "Statue of Khufu",
#     68: "Statue of King Ramesses ll - Goddess Anath",
#     69: "Statue of King Ramses II Grand Egyptian Museum",
#     70: "Statue of King Ramses II Luxor Temple",
#     71: "Statue of King Sety Il Holding Standards",
#     72: "Statue of King Zoser",
#     73: "Statue of Mentuhotep II",
#     74: "Statue of Merenptah as standard Bearer",
#     75: "Statue of Osiris",
#     76: "Statue of Queen Metnoforet",
#     77: "Statue of Ramesses III as standard Bearer",
#     78: "Statue of Snefru",
#     79: "Statue of Sobekhotep V",
#     80: "Statue of Tutankhamun",
#     81: "Stela of king Snefero",
#     82: "bust of Ramesses II",
#     83: "kneeling statue of queen hatshibsut",
# }
#
# # Confidence threshold for detection
# CONFIDENCE_THRESHOLD = 0.5


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""

    # ── Validate required secrets ─────────────────────────────────────────────
    missing: list[str] = []
    if not settings.hf_token:
        missing.append("HF_TOKEN")
    if not settings.mongodb_uri:
        missing.append("MONGODB_URI")
    if not settings.sbg_api_key:
        missing.append("SBG_API_KEY")
    if not settings.sbg_base_url:
        missing.append("SBG_BASE_URL")
    if missing:
        logger.critical(
            "Missing required environment variables: %s. "
            "Set them in .env or your environment and restart.",
            ", ".join(missing),
        )
        raise RuntimeError(
            f"Missing required config: {', '.join(missing)}"
        )

    # # ── Chonkie SemanticChunker [DISABLED for Ingestion] ────────────────────
    # logger.info(
    #     "Initializing Chonkie SemanticChunker with model '%s'...",
    #     settings.chunker_embedding_model,
    # )
    # app_state.chunker = create_chonkie_semantic_chunker(
    #     embedding_model_name=settings.chunker_embedding_model,
    #     similarity_level=settings.similarity_level,
    #     min_chunk_chars=settings.min_chunk_chars,
    #     max_chunk_chars=settings.max_chunk_chars,
    # )
    # logger.info("Chonkie SemanticChunker ready.")

    # ── HuggingFace InferenceClient ───────────────────────────────────────────
    logger.info(
        "Initializing HuggingFace InferenceClient (model: %s)...",
        settings.hf_embedding_model,
    )
    app_state.hf_client = InferenceClient(
        provider="hf-inference",
        api_key=settings.hf_token,
    )
    logger.info("HuggingFace InferenceClient ready.")

    # ── MongoDB ───────────────────────────────────────────────────────────────
    logger.info("Connecting to MongoDB...")
    app_state.mongo_client = create_motor_client(settings.mongodb_uri)
    app_state.collection = app_state.mongo_client[settings.mongodb_db][
        settings.mongodb_collection
    ]
    # await ensure_indexes(app_state.collection)
    logger.info(
        "MongoDB connected → %s / %s",
        settings.mongodb_db,
        settings.mongodb_collection,
    )

    # # ── YOLO Model ────────────────────────────────────────────────────────────
    # logger.info("Loading YOLO model for artifact detection...")
    # try:
    #     if YOLO_MODEL_PATH.exists():
    #         app_state.yolo_model = YOLO(str(YOLO_MODEL_PATH))
    #         logger.info(f"✅ YOLO model loaded successfully from {YOLO_MODEL_PATH}")
    #     else:
    #         logger.warning(
    #             f"⚠️  YOLO model not found at {YOLO_MODEL_PATH}. "
    #             "Artifact detection endpoints will return 503."
    #         )
    #         app_state.yolo_model = None
    # except Exception as exc:
    #     logger.error(f"❌ Failed to load YOLO model: {exc}")
    #     app_state.yolo_model = None

    logger.info("🚀 UEE Ingestion API is ready.")

    yield  # ── app is running ──────────────────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────────
    if app_state.mongo_client:
        app_state.mongo_client.close()
        logger.info("MongoDB connection closed.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="UEE Ingestion & Conversation API",
    description=(
        "Upload PDF articles for ingestion into a RAG knowledge base, "
        "then query that knowledge base with artifact context to get "
        "AI-powered answers grounded in the UEE corpus."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "API is running"}


@app.get("/health", tags=["Meta"])
async def health():
    """Liveness probe."""
    return {
        "status": "healthy",
        # "yolo_model_loaded": app_state.yolo_model is not None,
    }


# # ───────────────────────────────────────────────────────────────────────────────
# # Artifact Detection Endpoints (YOLO) - [DISABLED / COMMENTED OUT]
# # ───────────────────────────────────────────────────────────────────────────────
#
# # @app.post("/detect-artifact", tags=["Artifact Detection"])
# # async def detect_artifact(file: UploadFile = File(...)):
# #     """
# #     Detect Egyptian artifacts in uploaded images using YOLO
# #
# #     Args:
# #         file: Image file (JPG, PNG, etc.)
# #
# #     Returns:
# #         JSON with artifact_id (artifact name) and confidence score
# #     """
# #
# #     # Validate model is loaded
# #     if app_state.yolo_model is None:
# #         raise HTTPException(
# #             status_code=503,
# #             detail="YOLO model not loaded. Please check server configuration.",
# #         )
# #
# #     # Validate file type
# #     if not file.content_type.startswith("image/"):
# #         raise HTTPException(
# #             status_code=400, detail="Invalid file type. Please upload an image."
# #         )
# #
# #     try:
# #         # Read image file
# #         contents = await file.read()
# #         logger.info(f"📸 Received image: {file.filename} ({len(contents)} bytes)")
# #
# #         # Convert to numpy array
# #         nparr = np.frombuffer(contents, np.uint8)
# #         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# #
# #         if image is None:
# #             raise HTTPException(
# #                 status_code=400,
# #                 detail="Failed to decode image. Please upload a valid image file.",
# #             )
# #
# #         logger.info(f"🔍 Running inference on image shape: {image.shape}")
# #
# #         # Run YOLO inference
# #         results = app_state.yolo_model(image, conf=CONFIDENCE_THRESHOLD)
# #
# #         # Process results
# #         if len(results) == 0 or len(results[0].boxes) == 0:
# #             logger.info("❌ No artifacts detected in image")
# #             return {
# #                 "artifact_id": None,
# #                 "confidence": 0.0,
# #                 "message": "No artifact detected in image. Try a clearer image or different angle.",
# #             }
# #
# #         # Get the detection with highest confidence
# #         boxes = results[0].boxes
# #         confidences = boxes.conf.cpu().numpy()
# #         classes = boxes.cls.cpu().numpy().astype(int)
# #
# #         logger.info(f"📊 Found {len(boxes)} detection(s)")
# #
# #         # Find best detection
# #         best_idx = np.argmax(confidences)
# #         best_confidence = float(confidences[best_idx])
# #         best_class = int(classes[best_idx])
# #
# #         # Map class to artifact name
# #         artifact_name = ARTIFACT_MAPPING.get(
# #             best_class, f"Unknown Artifact (Class {best_class})"
# #         )
# #
# #         logger.info(f"✅ Detected: {artifact_name} (confidence: {best_confidence:.2%})")
# #
# #         return {
# #             "artifact_id": artifact_name,
# #             "confidence": best_confidence,
# #             "class_id": best_class,
# #             "detections_count": len(boxes),
# #         }
# #
# #     except HTTPException:
# #         raise
# #     except Exception as e:
# #         logger.error(f"❌ Detection failed: {str(e)}")
# #         import traceback
# #
# #         logger.error(traceback.format_exc())
# #         raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
# #
# #
# # @app.post("/detect-artifact-detailed", tags=["Artifact Detection"])
# # async def detect_artifact_detailed(file: UploadFile = File(...)):
# #     """
# #     Detect artifacts with detailed information including all detections and bounding boxes
# #
# #     Returns:
# #         JSON with all detected artifacts, their bounding boxes, and confidence scores
# #     """
# #
# #     if app_state.yolo_model is None:
# #         raise HTTPException(status_code=503, detail="Model not loaded")
# #
# #     try:
# #         # Read and decode image
# #         contents = await file.read()
# #         nparr = np.frombuffer(contents, np.uint8)
# #         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# #
# #         if image is None:
# #             raise HTTPException(status_code=400, detail="Invalid image")
# #
# #         # Run inference
# #         results = app_state.yolo_model(image, conf=CONFIDENCE_THRESHOLD)
# #
# #         if len(results) == 0 or len(results[0].boxes) == 0:
# #             return {"detections": [], "count": 0, "message": "No artifacts detected"}
# #
# #         # Extract all detections
# #         boxes = results[0].boxes
# #         detections = []
# #
# #         for i in range(len(boxes)):
# #             box = boxes[i]
# #             class_id = int(box.cls.cpu().numpy()[0])
# #             confidence = float(box.conf.cpu().numpy()[0])
# #             bbox = box.xyxy.cpu().numpy()[0].tolist()  # [x1, y1, x2, y2]
# #
# #             artifact_name = ARTIFACT_MAPPING.get(
# #                 class_id, f"Unknown Artifact (Class {class_id})"
# #             )
# #
# #             detections.append(
# #                 {
# #                     "artifact_id": artifact_name,
# #                     "confidence": confidence,
# #                     "class_id": class_id,
# #                     "bbox": {
# #                         "x1": float(bbox[0]),
# #                         "y1": float(bbox[1]),
# #                         "x2": float(bbox[2]),
# #                         "y2": float(bbox[3]),
# #                     },
# #                 }
# #             )
# #
# #         # Sort by confidence (highest first)
# #         detections.sort(key=lambda x: x["confidence"], reverse=True)
# #
# #         logger.info(f"✅ Returned {len(detections)} detection(s)")
# #
# #         return {
# #             "detections": detections,
# #             "count": len(detections),
# #             "image_shape": {
# #                 "height": image.shape[0],
# #                 "width": image.shape[1],
# #                 "channels": image.shape[2],
# #             },
# #         }
# #
# #     except HTTPException:
# #         raise
# #     except Exception as e:
# #         logger.error(f"Detailed detection failed: {str(e)}")
# #         import traceback
# #
# #         logger.error(traceback.format_exc())
# #         raise HTTPException(status_code=500, detail=str(e))
# #
# #
# # @app.get("/model-info", tags=["Artifact Detection"])
# # async def model_info():
# #     """Get information about the loaded model and available artifacts"""
# #     if app_state.yolo_model is None:
# #         return {"loaded": False, "error": "Model not loaded"}
# #
# #     return {
# #         "loaded": True,
# #         "model_path": str(YOLO_MODEL_PATH),
# #         "num_classes": len(ARTIFACT_MAPPING),
# #         "confidence_threshold": CONFIDENCE_THRESHOLD,
# #         "model_type": str(type(app_state.yolo_model).__name__),
# #         "sample_artifacts": list(ARTIFACT_MAPPING.values())[:10],  # Show first 10
# #         "total_artifacts": len(ARTIFACT_MAPPING),
# #     }
# #
# #
# # @app.get("/artifacts", tags=["Artifact Detection"])
# # async def list_artifacts():
# #     """Get complete list of all detectable artifacts"""
# #     artifacts_list = [
# #         {
# #             "class_id": class_id,
# #             "artifact_name": artifact_name,
# #         }
# #         for class_id, artifact_name in ARTIFACT_MAPPING.items()
# #     ]
# #
# #     return {
# #         "artifacts": artifacts_list,
# #         "total": len(artifacts_list),
# #     }
# #
# #
# # @app.get("/artifacts/{class_id}", tags=["Artifact Detection"])
# # async def get_artifact(class_id: int):
# #     """Get information about a specific artifact by class ID"""
# #     if class_id not in ARTIFACT_MAPPING:
# #         raise HTTPException(
# #             status_code=404, detail=f"Artifact with class_id {class_id} not found"
# #         )
# #
# #     return {
# #         "class_id": class_id,
# #         "artifact_name": ARTIFACT_MAPPING[class_id],
# #     }


# # ───────────────────────────────────────────────────────────────────────────────
# # Ingestion Endpoint - [DISABLED / COMMENTED OUT]
# # ───────────────────────────────────────────────────────────────────────────────
#
# # @app.post("/ingest", tags=["Ingestion"])
# # async def ingest(file: UploadFile = File(...)):
# #     """
# #     Ingest a single PDF file.
# #
# #     Steps:
# #       1. Validate the upload is a PDF.
# #       2. Check for duplicate article (reject if already in DB).
# #       3. Extract text + semantic chunks via Chonkie.
# #       4. Generate HuggingFace embeddings for each chunk (max 4 concurrent calls).
# #       5. Insert chunks (with embeddings) into MongoDB.
# #
# #     Returns a summary of what was stored.
# #     """
# #
# #     # ── 1. Validate content type ──────────────────────────────────────────────
# #     if file.content_type not in ("application/pdf", "application/octet-stream"):
# #         # Also accept by extension in case content-type is generic.
# #         if not (file.filename or "").lower().endswith(".pdf"):
# #             raise HTTPException(
# #                 status_code=status.HTTP_400_BAD_REQUEST,
# #                 detail="Only PDF files are accepted. Please upload a .pdf file.",
# #             )
# #
# #     file_name = file.filename or "upload.pdf"
# #     logger.info("Received upload: '%s'", file_name)
# #
# #     # ── 2. Read bytes ─────────────────────────────────────────────────────────
# #     try:
# #         pdf_bytes = await file.read()
# #     except Exception as exc:
# #         logger.error("Failed to read uploaded file: %s", exc)
# #         raise HTTPException(
# #             status_code=status.HTTP_400_BAD_REQUEST,
# #             detail=f"Could not read uploaded file: {exc}",
# #         )
# #
# #     if not pdf_bytes:
# #         raise HTTPException(
# #             status_code=status.HTTP_400_BAD_REQUEST,
# #             detail="Uploaded file is empty.",
# #         )
# #
# #     # ── 3. Extract chunks (CPU-bound, runs in thread pool) ────────────────────
# #     try:
# #         article, chunks = await extract_chunks(
# #             pdf_bytes=pdf_bytes,
# #             file_name=file_name,
# #             chunker=app_state.chunker,
# #             min_chunk_chars=settings.min_chunk_chars,
# #             chunk_scope=settings.chunk_scope,
# #         )
# #     except Exception as exc:
# #         logger.error("Extraction failed for '%s': %s", file_name, exc)
# #         raise HTTPException(
# #             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
# #             detail=f"PDF extraction failed: {exc}",
# #         )
# #
# #     article_id = article["article_id"]
# #     logger.info(
# #         "'%s' → article_id=%s, %d chunks extracted",
# #         file_name,
# #         article_id,
# #         len(chunks),
# #     )
# #
# #     # ── 4. Reject duplicates ───────────────────────────────────────────────────
# #     try:
# #         already_exists = await check_article_exists(
# #             app_state.collection, article_id
# #         )
# #     except Exception as exc:
# #         logger.error("MongoDB duplicate check failed: %s", exc)
# #         raise HTTPException(
# #             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
# #             detail=f"Database error during duplicate check: {exc}",
# #         )
# #
# #     if already_exists:
# #         raise HTTPException(
# #             status_code=status.HTTP_409_CONFLICT,
# #             detail=(
# #                 f"Article '{file_name}' (id: {article_id}) has already been ingested. "
# #                 "Duplicate uploads are not allowed. "
# #                 "If you need to re-ingest, delete the existing records first."
# #             ),
# #         )
# #
# #     # ── 5. Embed chunks ────────────────────────────────────────────────────────
# #     try:
# #         chunks = await embed_chunks(
# #             chunks=chunks,
# #             client=app_state.hf_client,
# #             model=settings.hf_embedding_model,
# #             concurrency=settings.hf_embedding_concurrency,
# #         )
# #     except Exception as exc:
# #         logger.error("Embedding failed for '%s': %s", file_name, exc)
# #         raise HTTPException(
# #             status_code=status.HTTP_502_BAD_GATEWAY,
# #             detail=f"HuggingFace embedding error: {exc}",
# #         )
# #
# #     logger.info(
# #         "Embeddings generated for %d chunks of '%s'.", len(chunks), file_name
# #     )
# #
# #     # ── 6. Store in MongoDB ────────────────────────────────────────────────────
# #     try:
# #         stored_count = await insert_chunks(app_state.collection, chunks)
# #     except Exception as exc:
# #         logger.error("MongoDB insert failed for '%s': %s", file_name, exc)
# #         raise HTTPException(
# #             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
# #             detail=f"Database insert error: {exc}",
# #         )
# #
# #     logger.info(
# #         "Stored %d / %d chunks for '%s'.", stored_count, len(chunks), file_name
# #     )
# #
# #     return JSONResponse(
# #         status_code=status.HTTP_200_OK,
# #         content={
# #             "status": "ok",
# #             "file_name": file_name,
# #             "article_id": article_id,
# #             "title": article.get("title", ""),
# #             "page_count": article.get("page_count"),
# #             "chunks_extracted": len(chunks),
# #             "chunks_stored": stored_count,
# #         },
# #     )


# ───────────────────────────────────────────────────────────────────────────────
# Conversation endpoint
# ───────────────────────────────────────────────────────────────────────────────

@app.post("/conversation", response_model=ConversationResponse, tags=["Conversation"])
async def conversation(body: ConversationRequest):
    """
    Answer a question about an artifact using RAG over the UEE knowledge base.

    Steps:
      1. Build a combined query string from artifact fields + user question.
      2. Embed the query via HuggingFace (same model used at ingest time).
      3. Run MongoDB Atlas $vectorSearch to retrieve top-K relevant chunks.
      4. Build system prompt (defines assistant behavior/tone).
      5. Build user message content (question + artifact data + retrieved passages).
      6. Build message list: conversation history + final user message.
      7. Call the SBG chat API.
      8. Return the model's answer together with the source chunks.
    """

    # ── 1. Build query text ───────────────────────────────────────────────────────
    query_text = build_query_text(
        artifact=body.artifact,
        question=body.question,
    )
    logger.info(
        "Conversation request | artifact='%s' | query='%s'",
        body.artifact.name,
        query_text[:120],
    )

    # ── 2. Embed the query ───────────────────────────────────────────────────────
    # Reuse the same embedder helper with a single-element list.
    try:
        query_chunk = [{"text": query_text, "chunk_id": "__query__"}]
        await embed_chunks(
            chunks=query_chunk,
            client=app_state.hf_client,
            model=settings.hf_embedding_model,
            concurrency=1,
        )
        query_embedding: list[float] = query_chunk[0].get("embedding", [])
    except Exception as exc:
        logger.error("Query embedding failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to embed query: {exc}",
        )

    if not query_embedding:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="HuggingFace returned an empty embedding for the query.",
        )

    # ── 3. Vector search ─────────────────────────────────────────────────────────
    try:
        retrieved = await vector_search(
            collection=app_state.collection,
            query_embedding=query_embedding,
            index_name=settings.mongodb_vector_index,
            top_k=settings.vector_search_top_k,
        )
    except Exception as exc:
        logger.error("Vector search failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector search failed: {exc}",
        )

    logger.info("Retrieved %d chunks from vector DB.", len(retrieved))

    # ── 4. Build system prompt (behavior/tone only) ───────────────────────────────
    system_prompt = build_system_prompt()

    # ── 5. Build user message content (question + artifact + passages) ────────────
    user_message_content = build_user_message(
        artifact=body.artifact,
        question=body.question,
        chunks=retrieved,
    )

    # ── 6. Build message list for the LLM ──────────────────────────────────────────
    # Include the full conversation history from the request body,
    # then append the current question (with artifact data and passages) as the final user turn.
    llm_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in body.messages
    ]
    llm_messages.append({"role": "user", "content": user_message_content})

    # ── 7. Call SBG chat API ──────────────────────────────────────────────────────
    raw_response = await call_chat(
        base_url=settings.sbg_base_url,
        api_key=settings.sbg_api_key,
        model_id=settings.llm_model_id,
        system_prompt=system_prompt,
        messages=llm_messages,
        max_tokens=settings.llm_max_tokens,
    )

    # Extract structured response (dict with output_text + metadata)
    llm_data = extract_answer(raw_response)
    answer = llm_data.get("output_text", "")

    logger.info(
        "Conversation complete | model=%s | chunks_used=%d | answer_chars=%d",
        settings.llm_model_id,
        len(retrieved),
        len(answer),
    )

    # ── 8. Build and return response ───────────────────────────────────────────────
    sources = [
        RetrievedChunk(
            chunk_id=c.get("chunk_id"),
            title=c.get("title"),
            text=c.get("text"),
            page_start=c.get("page_start"),
            page_end=c.get("page_end"),
            score=c.get("score"),
        )
        for c in retrieved
    ]

    # Prepare additional metadata (all fields except output_text)
    additional_metadata = {
        k: v for k, v in llm_data.items() 
        if k not in ("output_text", "model_id")  # Already in response
    }

    return ConversationResponse(
        answer=answer,
        retrieved_chunks=len(retrieved),
        model_id=llm_data.get("model_id", settings.llm_model_id),
        sources=sources,
        request_id=llm_data.get("request_id"),
        region=llm_data.get("region"),
        usage=llm_data.get("usage"),
        estimated_cost_usd=llm_data.get("estimated_cost_usd"),
        actual_cost_usd=llm_data.get("actual_cost_usd"),
        status=llm_data.get("status"),
        llm_response_metadata=additional_metadata if additional_metadata else None,
    )
