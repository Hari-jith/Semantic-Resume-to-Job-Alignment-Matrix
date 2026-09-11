"""
FastAPI application entry point.

Run from the backend/ directory with:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import health
from app.ml.embedding_model import embedding_model

settings = get_settings()

app = FastAPI(
    title="Semantic Resume ATS API",
    description="AI-powered resume analysis and job matching backend.",
    version="0.1.0",
)


@app.on_event("startup")
def load_ml_models() -> None:
    """
    Loads the embedding model once when the server starts, not per-request.
    If no fine-tuned model is present, this loads the pretrained fallback
    and logs a warning — the app still starts and works either way.
    """
    embedding_model.load()

# Allow the local Vite dev server (and configured frontend origin) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)

# More routers (resume, matching, recommendations) are added here in later modules.
