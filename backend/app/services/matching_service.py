"""
Resume + JD pipeline: orchestrates Modules 2-7 for the matching endpoints.
Operates on already-extracted resume/JD text — parsing/validation happens
in the route layer (api/routes/matching.py).
"""

from app.ml.embedding_model import embedding_model
from app.ml.skill_extractor import get_skill_extractor
from app.schemas.match_schemas import (
    MatchAnalysisResponse,
    RecommendImprovementsResponse,
)
from app.schemas.resume_schemas import ScoreComponent
from app.services.ats_service import score_resume_jd_match
from app.services.groq_service import get_groq_service

# Deterministic strength statements, each tied to a specific score component
# scoring 70+. This keeps "Strengths" explainable and grounded in the actual
# computed score rather than invented or LLM-generated praise.
_STRENGTH_THRESHOLD = 70.0
_STRENGTH_MESSAGES = {
    "semantic_similarity": "The overall content and phrasing of your resume closely aligns with this job description.",
    "skills_match": "Your resume covers a strong portion of the skills mentioned in this job description.",
    "keyword_relevance": "Your resume uses much of the same terminology as the job description.",
    "experience_alignment": "Your experience duration appears to align well with what this role expects.",
    "resume_completeness": "Your resume includes the key sections expected by most ATS systems.",
}


def _derive_strengths(breakdown_raw: dict) -> list[str]:
    strengths = []
    for component_name, message in _STRENGTH_MESSAGES.items():
        component = breakdown_raw.get(component_name)
        if component and component["score"] >= _STRENGTH_THRESHOLD:
            strengths.append(message)
    return strengths


def analyze_match(resume_text: str, jd_text: str) -> MatchAnalysisResponse:
    skill_extractor = get_skill_extractor()
    resume_skills = skill_extractor.extract_names(resume_text)
    jd_skills = skill_extractor.extract_names(jd_text)

    resume_embedding = embedding_model.encode(resume_text)
    jd_embedding = embedding_model.encode(jd_text)

    ats_result = score_resume_jd_match(
        resume_text, jd_text, resume_embedding, jd_embedding, resume_skills, jd_skills,
    )
    breakdown_raw = dict(ats_result["breakdown"])
    excluded = breakdown_raw.pop("_excluded_components", [])
    score_breakdown = {name: ScoreComponent(**comp) for name, comp in breakdown_raw.items()}

    strengths = _derive_strengths(breakdown_raw)

    resume_skill_set = set(resume_skills)
    jd_skill_set = set(jd_skills)
    matching_skills = sorted(resume_skill_set & jd_skill_set)
    missing_skills = sorted(jd_skill_set - resume_skill_set)

    groq_service = get_groq_service()
    jd_analysis = groq_service.analyze_job_description(jd_text, deterministic_skills=jd_skills)

    return MatchAnalysisResponse(
        ats_match_score=ats_result["overall_score"],
        score_breakdown=score_breakdown,
        excluded_score_components=excluded,
        score_notes=ats_result["notes"],
        matching_skills=matching_skills,
        missing_skills=missing_skills,
        strengths=strengths,
        jd_job_title=jd_analysis["job_title"],
        jd_required_skills=jd_analysis["required_skills"],
        jd_preferred_skills=jd_analysis["preferred_skills"],
        jd_analysis_source=jd_analysis["source"],
        embedding_model_type=embedding_model.get_model_info()["model_type"],
        resume_text=resume_text,
        jd_text=jd_text,
    )


def recommend_improvements(resume_text: str, jd_text: str) -> RecommendImprovementsResponse:
    skill_extractor = get_skill_extractor()
    resume_skills = set(skill_extractor.extract_names(resume_text))
    jd_skills = set(skill_extractor.extract_names(jd_text))

    matching_skills = sorted(resume_skills & jd_skills)
    missing_skills = sorted(jd_skills - resume_skills)

    groq_service = get_groq_service()
    result = groq_service.generate_improvements(resume_text, jd_text, matching_skills, missing_skills)

    return RecommendImprovementsResponse(
        critical_improvements=result["critical_improvements"],
        recommended_improvements=result["recommended_improvements"],
        optional_improvements=result["optional_improvements"],
        source=result["source"],
    )
