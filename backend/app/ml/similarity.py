"""
Similarity utilities for comparing embeddings.

Kept separate from embedding_model.py so scoring logic (ats_service,
role_matcher) depends on plain numpy arrays, not on the SentenceTransformer
class itself.
"""

import numpy as np


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Returns cosine similarity between two vectors, as a float in [-1, 1].
    In practice, sentence embeddings from this model tend to fall in a much
    narrower positive range, but the raw value is returned unclamped here —
    callers decide how to interpret/scale it (see ats_service for how this
    gets mapped to a 0-100 score component).
    """
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def cosine_similarity_batch(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compares one vector against many (e.g. one resume against all role
    profile embeddings). Returns an array of similarity scores, same order
    as the rows of `matrix`.
    """
    query_norm = np.linalg.norm(query_vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    # Avoid division by zero for any degenerate rows.
    safe_matrix_norms = np.where(matrix_norms == 0, 1e-10, matrix_norms)
    safe_query_norm = query_norm if query_norm != 0 else 1e-10

    similarities = (matrix @ query_vector) / (safe_matrix_norms * safe_query_norm)
    return similarities
