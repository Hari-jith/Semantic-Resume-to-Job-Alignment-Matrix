"""
Schemas for /analyze/resume and /recommend/jobs.
"""

from pydantic import BaseModel, Field


class ScoreComponent(BaseModel):
    score: float = Field(..., description="0-100 score for this component.")
    weight: float = Field(..., description="This component's weight in the overall score, after any renormalization.")


class ResumeAnalysisResponse(BaseModel):
    ats_score: float = Field(..., description="Overall general resume-quality score, 0-100.")
    score_breakdown: dict[str, ScoreComponent]
    excluded_score_components: list[str] = Field(
        default_factory=list,
        description="Components that could not be computed for this resume and were excluded from the score.",
    )
    score_notes: str = Field(..., description="Plain-language caveat about what this score does and does not mean.")

    summary: str
    detected_skills: list[str]
    skills_by_category: dict[str, list[str]]
    experience_summary: str
    education_summary: str
    strengths: list[str]
    areas_for_improvement: list[str]

    analysis_source: str = Field(..., description="'groq' or 'fallback' — which path produced the narrative fields above.")
    embedding_model_type: str = Field(..., description="'fine_tuned' or 'pretrained_fallback'.")

    resume_text: str = Field(
        ...,
        description="The extracted resume text, returned so the frontend can pass it to /recommend/jobs "
                    "without re-uploading the file.",
    )


class RecommendJobsRequest(BaseModel):
    resume_text: str = Field(..., min_length=40, description="Resume text, as returned by /analyze/resume.")
    top_n: int = Field(default=5, ge=1, le=16, description="How many roles to return.")


class RecommendedRole(BaseModel):
    role_name: str
    match_score: float = Field(..., description="Combined semantic + skill-overlap score, 0-100.")
    semantic_similarity: float = Field(..., description="Semantic similarity component, 0-100.")
    skill_match_score: float = Field(..., description="Skill overlap component, 0-100.")
    matching_skills: list[str]
    explanation: str


class RecommendJobsResponse(BaseModel):
    recommended_roles: list[RecommendedRole]
    embedding_model_type: str
    explanation_source: str = Field(..., description="'groq' or 'fallback'.")
