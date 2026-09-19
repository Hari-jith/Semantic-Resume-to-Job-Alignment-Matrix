"""
Role matching for the Resume-only "Recommended Jobs" feature.

Pipeline:
    resume text + resume skills
        -> semantic similarity against each role profile's embedding
        -> skill overlap against each role profile's common_skills
        -> weighted combination (see config.role_match_semantic_weight / _skill_weight)
        -> ranked list of roles

Design notes:
- Skill overlap is weighted higher than semantic similarity by default (see
  config.py comment) because it's directly interpretable and doesn't depend
  on how well the embedding model actually captures resume/job semantics —
  which, with only a pretrained (not fine-tuned) model in use so far, is
  unproven for this specific task.
- Role embeddings are computed once and cached in memory, not per request.
- This module returns raw ranked data (scores + matching skills) — it does
  NOT generate the natural-language "why this role fits" explanation.
  That's produced later (recommendation_service + groq_service) and is
  grounded in the matching_skills list returned here, not invented freely.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.ml.embedding_model import embedding_model
from app.ml.similarity import cosine_similarity_batch

logger = logging.getLogger(__name__)
settings = get_settings()

_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "role_profiles.json"


class RoleMatcher:
    def __init__(self, role_profiles_path: Path = _DATA_PATH) -> None:
        with open(role_profiles_path, "r", encoding="utf-8") as f:
            self.roles: list[dict] = json.load(f)["roles"]

        self._role_embeddings: np.ndarray | None = None

    @staticmethod
    def _role_text(role: dict) -> str:
        """
        Builds the text used to embed a role profile: description + skills +
        responsibilities combined, so the embedding captures the full profile
        rather than just the short description.
        """
        skills_text = ", ".join(role["common_skills"])
        responsibilities_text = "; ".join(role["typical_responsibilities"])
        return (
            f"{role['name']}. {role['description']} "
            f"Key skills: {skills_text}. "
            f"Responsibilities: {responsibilities_text}."
        )

    def _ensure_role_embeddings(self) -> None:
        """Computes role embeddings lazily, once, on first use."""
        if self._role_embeddings is not None:
            return

        if not embedding_model.is_loaded():
            raise RuntimeError(
                "Embedding model is not loaded yet. Role matching requires "
                "embedding_model.load() to have run at application startup."
            )

        texts = [self._role_text(role) for role in self.roles]
        self._role_embeddings = embedding_model.encode_batch(texts)
        logger.info(f"Computed embeddings for {len(self.roles)} role profiles.")

    def rank_roles(
        self,
        resume_text: str,
        resume_skills: list[str],
        top_n: int = 5,
    ) -> list[dict]:
        """
        Returns the top_n roles ranked by combined semantic + skill-overlap score.

        Each result:
        {
            "role_name": str,
            "description": str,
            "match_score": float (0-100),
            "semantic_similarity": float (0-100),   # component, for transparency
            "skill_match_score": float (0-100),     # component, for transparency
            "matching_skills": list[str],           # grounds the later LLM explanation
        }
        """
        if not resume_text or not resume_text.strip():
            raise ValueError("Cannot rank roles: resume text is empty.")

        self._ensure_role_embeddings()

        resume_embedding = embedding_model.encode(resume_text)
        semantic_sims = cosine_similarity_batch(resume_embedding, self._role_embeddings)

        resume_skill_set = set(resume_skills)
        results = []

        for i, role in enumerate(self.roles):
            role_skill_set = set(role["common_skills"])

            # Skill score = what fraction of THIS ROLE's expected skills the
            # resume already covers. Using the role's skill count as the
            # denominator (not the resume's) keeps the score meaningful even
            # for resumes with many skills unrelated to this particular role.
            matching_skills = sorted(resume_skill_set & role_skill_set)
            skill_score = len(matching_skills) / len(role_skill_set) if role_skill_set else 0.0

            # Cosine similarity from a sentence embedding model is typically
            # positive in practice for related text, but clip defensively —
            # a negative contribution here would be uninterpretable to a user.
            semantic_score = max(0.0, float(semantic_sims[i]))

            combined_score = (
                settings.role_match_semantic_weight * semantic_score
                + settings.role_match_skill_weight * skill_score
            )

            results.append({
                "role_name": role["name"],
                "description": role["description"],
                "match_score": round(combined_score * 100, 1),
                "semantic_similarity": round(semantic_score * 100, 1),
                "skill_match_score": round(skill_score * 100, 1),
                "matching_skills": matching_skills,
            })

        results.sort(key=lambda r: r["match_score"], reverse=True)
        return results[:top_n]


@lru_cache
def get_role_matcher() -> RoleMatcher:
    """Cached accessor — loads role_profiles.json once per process."""
    return RoleMatcher()
