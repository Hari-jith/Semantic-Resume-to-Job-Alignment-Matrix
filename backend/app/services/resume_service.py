"""
Resume-only pipeline: orchestrates Modules 2-7 into the two resume-only
endpoints' logic. Operates on already-extracted resume text — file
parsing/validation happens in the route layer (api/routes/resume.py), so
these functions are directly testable with plain strings.
"""

from collections import defaultdict

from app.ml.embedding_model import embedding_model
from app.ml.role_matcher import get_role_matcher
from app.ml.skill_extractor import get_skill_extractor
from app.schemas.resume_schemas import (
    RecommendedRole,
    RecommendJobsResponse,
    ResumeAnalysisResponse,
    ScoreComponent,
)
from app.services.ats_service import score_resume_only
from app.services.groq_service import get_groq_service


def analyze_resume(resume_text: str) -> ResumeAnalysisResponse:
    """
    Full resume-only analysis: skill extraction, role-alignment signal,
    ATS scoring, and Groq-powered narrative summary (with fallback).
    """
    skill_extractor = get_skill_extractor()
    extracted_skills = skill_extractor.extract(resume_text)   # [{canonical, category}, ...]
    skill_names = [s["canonical"] for s in extracted_skills]

    skills_by_category: dict[str, list[str]] = defaultdict(list)
    for s in extracted_skills:
        skills_by_category[s["category"]].append(s["canonical"])

    # Role-alignment signal: reuse the top-ranked role's semantic similarity
    # as the embedding model's contribution to the overall ATS score.
    role_matcher = get_role_matcher()
    top_role_matches = role_matcher.rank_roles(resume_text, skill_names, top_n=1)
    top_role_semantic = top_role_matches[0]["semantic_similarity"] if top_role_matches else 0.0

    ats_result = score_resume_only(resume_text, skill_names, top_role_semantic)
    breakdown_raw = dict(ats_result["breakdown"])
    excluded = breakdown_raw.pop("_excluded_components", [])
    score_breakdown = {name: ScoreComponent(**comp) for name, comp in breakdown_raw.items()}

    groq_service = get_groq_service()
    groq_result = groq_service.analyze_resume(resume_text, deterministic_skills=skill_names)

    return ResumeAnalysisResponse(
        ats_score=ats_result["overall_score"],
        score_breakdown=score_breakdown,
        excluded_score_components=excluded,
        score_notes=ats_result["notes"],
        summary=groq_result["summary"],
        detected_skills=skill_names,
        skills_by_category=dict(skills_by_category),
        experience_summary=groq_result["experience_summary"],
        education_summary=groq_result["education_summary"],
        strengths=groq_result["strengths"],
        areas_for_improvement=groq_result["improvements"],
        analysis_source=groq_result["source"],
        embedding_model_type=embedding_model.get_model_info()["model_type"],
        resume_text=resume_text,
    )


def recommend_jobs(resume_text: str, top_n: int = 5) -> RecommendJobsResponse:
    """
    Ranks role profiles against the resume and attaches a grounded
    explanation (Groq, with fallback) per role.
    """
    skill_extractor = get_skill_extractor()
    skill_names = skill_extractor.extract_names(resume_text)

    role_matcher = get_role_matcher()
    ranked_roles = role_matcher.rank_roles(resume_text, skill_names, top_n=top_n)

    groq_service = get_groq_service()
    explanation_result = groq_service.generate_role_explanations(ranked_roles)
    explanations = explanation_result["explanations"]

    recommended = [
        RecommendedRole(
            role_name=role["role_name"],
            match_score=role["match_score"],
            semantic_similarity=role["semantic_similarity"],
            skill_match_score=role["skill_match_score"],
            matching_skills=role["matching_skills"],
            explanation=explanations.get(role["role_name"], ""),
        )
        for role in ranked_roles
    ]

    return RecommendJobsResponse(
        recommended_roles=recommended,
        embedding_model_type=embedding_model.get_model_info()["model_type"],
        explanation_source=explanation_result["source"],
    )
