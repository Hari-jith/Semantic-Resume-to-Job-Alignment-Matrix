"""
Schemas for /analyze/match and /recommend/improvements.
"""

from pydantic import BaseModel, Field

from app.schemas.resume_schemas import ScoreComponent


class MatchAnalysisResponse(BaseModel):
    ats_match_score: float = Field(..., description="ATS match score for this specific resume-JD pair, 0-100.")
    score_breakdown: dict[str, ScoreComponent]
    excluded_score_components: list[str] = Field(default_factory=list)
    score_notes: str

    matching_skills: list[str]
    missing_skills: list[str]
    strengths: list[str] = Field(
        ..., description="Deterministic statements grounded in score components that scored 70+."
    )

    jd_job_title: str
    jd_required_skills: list[str]
    jd_preferred_skills: list[str]
    jd_analysis_source: str = Field(..., description="'groq' or 'fallback'.")

    embedding_model_type: str
    resume_text: str = Field(..., description="Returned so the frontend can call /recommend/improvements without re-uploading.")
    jd_text: str = Field(..., description="Returned so the frontend can call /recommend/improvements without re-submitting.")


class RecommendImprovementsRequest(BaseModel):
    resume_text: str = Field(..., min_length=40)
    jd_text: str = Field(..., min_length=20)


class RecommendImprovementsResponse(BaseModel):
    critical_improvements: list[str]
    recommended_improvements: list[str]
    optional_improvements: list[str]
    source: str = Field(..., description="'groq' or 'fallback'.")
