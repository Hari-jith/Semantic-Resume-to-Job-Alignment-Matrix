"""
Health check endpoint.

Reports which embedding model is actually loaded (fine-tuned vs pretrained
fallback) so the frontend/README never have to guess or assume.
"""

from fastapi import APIRouter

from app.ml.embedding_model import embedding_model

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Semantic Resume ATS API",
        "embedding_model": embedding_model.get_model_info(),
    }
