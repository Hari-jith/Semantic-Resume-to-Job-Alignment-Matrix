"""
Embedding model loader.

Responsible for ONE thing: giving the rest of the app a loaded
SentenceTransformer model and a function to embed text with it.

Loading strategy:
  1. If backend/models/<finetuned_model_path> exists and looks like a valid
     saved SentenceTransformer model, load it. This is the fine-tuned
     resume-job matching model once it exists.
  2. Otherwise, fall back to the pretrained model from Hugging Face
     (sentence-transformers/all-mpnet-base-v2) and log a clear warning.

The app must never silently pretend the fine-tuned model is in use when it
isn't — every caller can check `get_model_info()` to know which model
actually produced a given embedding, and this flows through to API
responses so the frontend/README never misrepresent it.

The model is loaded once per process (singleton) — loading a transformer
model is expensive and should not happen per-request.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Resolved relative to backend/ (where uvicorn is run from).
_BACKEND_ROOT = Path(__file__).resolve().parents[2]


class EmbeddingModel:
    """Thin wrapper around a loaded SentenceTransformer instance."""

    def __init__(self) -> None:
        self._model: Optional[SentenceTransformer] = None
        self._model_type: str = "unloaded"   # "fine_tuned" | "pretrained_fallback"
        self._model_name: str = ""

    def load(self) -> None:
        """
        Loads the model. Called once at application startup.
        Safe to call again (e.g. in tests) — it just reloads.
        """
        finetuned_path = _BACKEND_ROOT / settings.finetuned_model_path

        if self._looks_like_valid_model_dir(finetuned_path):
            try:
                logger.info(f"Loading fine-tuned model from {finetuned_path}")
                self._model = SentenceTransformer(str(finetuned_path))
                self._model_type = "fine_tuned"
                self._model_name = str(finetuned_path)
                return
            except Exception as exc:
                # If the folder exists but is malformed/corrupted, don't crash
                # the whole app — fall back and log loudly so it's noticed.
                logger.warning(
                    f"Found a model folder at {finetuned_path} but failed to "
                    f"load it ({exc}). Falling back to the pretrained model."
                )

        logger.warning(
            f"No fine-tuned model found at {finetuned_path}. "
            f"Loading pretrained fallback: {settings.pretrained_model_name}. "
            f"Semantic matching quality reflects the general-purpose pretrained "
            f"model, not a resume/job fine-tuned one."
        )
        self._model = SentenceTransformer(settings.pretrained_model_name)
        self._model_type = "pretrained_fallback"
        self._model_name = settings.pretrained_model_name

    @staticmethod
    def _looks_like_valid_model_dir(path: Path) -> bool:
        """
        A saved SentenceTransformer directory always contains a
        config_sentence_transformers.json file. Checking for it avoids
        trying (and failing) to load an empty or unrelated folder.
        """
        return path.is_dir() and (path / "config_sentence_transformers.json").exists()

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model_info(self) -> dict:
        return {
            "model_type": self._model_type,
            "model_name": self._model_name,
        }

    def encode(self, text: str) -> np.ndarray:
        """
        Encodes a single string into a 768-dim embedding vector (numpy array).
        Raises RuntimeError if called before load().
        """
        if self._model is None:
            raise RuntimeError(
                "Embedding model has not been loaded yet. "
                "Call embedding_model.load() at application startup."
            )
        if not text or not text.strip():
            raise ValueError("Cannot encode empty text.")

        embedding = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encodes multiple strings at once. More efficient than calling encode()
        in a loop when embedding several role profiles, for example.
        """
        if self._model is None:
            raise RuntimeError(
                "Embedding model has not been loaded yet. "
                "Call embedding_model.load() at application startup."
            )
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


# Module-level singleton — imported and shared across the app.
embedding_model = EmbeddingModel()
