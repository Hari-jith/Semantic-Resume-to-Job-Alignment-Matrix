"""
Resume + JD endpoints: POST /analyze/match and POST /recommend/improvements.

/analyze/match accepts a resume file plus a job description supplied EITHER
as plain text (jd_text form field) OR as a file (jd_file) — not both required,
but at least one must be present, per the project's "JD is optional overall,
but flexible in format when provided" requirement.
"""

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.parsers.document_extractor import extract_text
from app.parsers.file_validator import FileValidationError, read_and_validate_upload
from app.schemas.match_schemas import (
    MatchAnalysisResponse,
    RecommendImprovementsRequest,
    RecommendImprovementsResponse,
)
from app.services import matching_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["matching"])

MIN_JD_TEXT_LENGTH = 20


@router.post("/analyze/match", response_model=MatchAnalysisResponse)
async def analyze_match_endpoint(
    resume_file: UploadFile = File(..., description="Resume as PDF or DOCX."),
    jd_text: str | None = Form(None, description="Job description as plain text."),
    jd_file: UploadFile | None = File(None, description="Job description as PDF or DOCX."),
) -> MatchAnalysisResponse:
    try:
        resume_bytes, resume_extension = await read_and_validate_upload(resume_file)
        resume_text = extract_text(resume_bytes, resume_extension)

        resolved_jd_text = await _resolve_jd_text(jd_text, jd_file)
    except FileValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        return matching_service.analyze_match(resume_text, resolved_jd_text)
    except RuntimeError as exc:
        logger.error(f"Match analysis unavailable: {exc}")
        raise HTTPException(status_code=503, detail="Analysis service is temporarily unavailable.")
    except Exception:
        logger.exception("Unexpected error during match analysis.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while analyzing the match.")


@router.post("/recommend/improvements", response_model=RecommendImprovementsResponse)
async def recommend_improvements_endpoint(
    payload: RecommendImprovementsRequest,
) -> RecommendImprovementsResponse:
    try:
        return matching_service.recommend_improvements(payload.resume_text, payload.jd_text)
    except RuntimeError as exc:
        logger.error(f"Improvement recommendations unavailable: {exc}")
        raise HTTPException(status_code=503, detail="Recommendation service is temporarily unavailable.")
    except Exception:
        logger.exception("Unexpected error during improvement recommendations.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating improvements.")


async def _resolve_jd_text(jd_text: str | None, jd_file: UploadFile | None) -> str:
    """
    Resolves the job description text from whichever source was provided.
    Prefers an uploaded JD file over pasted text if somehow both are sent
    (unusual, but a clear behavior is better than an ambiguous one).
    Raises FileValidationError with a clear message if neither is usable.
    """
    if jd_file is not None and jd_file.filename:
        jd_bytes, jd_extension = await read_and_validate_upload(jd_file)
        return extract_text(jd_bytes, jd_extension)

    if jd_text and jd_text.strip():
        cleaned = jd_text.strip()
        if len(cleaned) < MIN_JD_TEXT_LENGTH:
            raise FileValidationError(
                f"Job description text is too short (minimum {MIN_JD_TEXT_LENGTH} characters)."
            )
        return cleaned

    raise FileValidationError(
        "A job description is required for this endpoint — provide it as text or upload a file."
    )
