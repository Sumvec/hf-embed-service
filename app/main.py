import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from .settings import settings
from .model import embed_model
import numpy as np
from nltk.tokenize import word_tokenize

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("hf-embed-service")

app = FastAPI(
    title="HF Embedding Service",
    description="Generates vector embeddings via Sentence-Transformers (Hugging Face).",
    version="1.0.0",
)

class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dims: int

@app.on_event("startup")
def on_startup():
    if settings.TRANSFORMERS_OFFLINE:
        import os
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        log.info("TRANSFORMERS_OFFLINE enabled")
    embed_model.load()
    log.info("Model loaded and ready")

@app.get("/healthz", tags=["utility"])
def health():
    return {"status": "ok"}

@app.post("/embed", response_model=EmbedResponse, tags=["embed"])
async def embed(req: EmbedRequest):
    try:
        texts = req.texts
        all_embeddings = []

        for text in texts:
            # Split long text into overlapping chunks
            chunks = chunk_text(text)

            # Encode each chunk and average into one vector
            chunk_vectors = embed_model.encode(chunks)
            avg_vector = np.mean(chunk_vectors, axis=0)
            all_embeddings.append(list(map(float, avg_vector.tolist())))

        dims = len(all_embeddings[0]) if all_embeddings else 0
        return {"embeddings": all_embeddings, "dims": dims}

    except Exception as e:
        log.exception("Embedding failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info", tags=["utility"])
def info():
    return {"model": settings.MODEL_NAME, "batch_size": settings.BATCH_SIZE}

# Helper: Chunk long texts
def chunk_text(text: str, max_tokens: int = 250, overlap: int = 30):
    tokens = word_tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = tokens[start:end]
        chunks.append(" ".join(chunk))
        start += max_tokens - overlap
    return chunks or [""]


