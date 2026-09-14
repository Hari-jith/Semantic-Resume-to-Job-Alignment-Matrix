"""
Resume-only endpoints: POST /analyze/resume and POST /recommend/jobs.

File validation and text extraction happen here (I/O concerns); the actual
analysis logic lives in resume_service, operating on plain text.
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.parsers.document_extractor import extract_text
from app.parsers.file_validator import FileValidationError, read_and_validate_upload
from app.schemas.resume_schemas import (
    RecommendJobsRequest,
    RecommendJobsResponse,
    ResumeAnalysisResponse,
)
from app.services import resume_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["resume"])


@router.post("/analyze/resume", response_model=ResumeAnalysisResponse)
async def analyze_resume_endpoint(file: UploadFile = File(...)) -> ResumeAnalysisResponse:
    try:
        file_bytes, extension = await read_and_validate_upload(file)
        resume_text = extract_text(file_bytes, extension)
    except FileValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        return resume_service.analyze_resume(resume_text)
    except RuntimeError as exc:
        # e.g. embedding model not loaded — a server-side setup problem, not a bad request
        logger.error(f"Resume analysis unavailable: {exc}")
        raise HTTPException(status_code=503, detail="Analysis service is temporarily unavailable.")
    except Exception:
        logger.exception("Unexpected error during resume analysis.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while analyzing the resume.")


@router.post("/recommend/jobs", response_model=RecommendJobsResponse)
async def recommend_jobs_endpoint(payload: RecommendJobsRequest) -> RecommendJobsResponse:
    try:
        return resume_service.recommend_jobs(payload.resume_text, top_n=payload.top_n)
    except RuntimeError as exc:
        logger.error(f"Job recommendation unavailable: {exc}")
        raise HTTPException(status_code=503, detail="Recommendation service is temporarily unavailable.")
    except Exception:
        logger.exception("Unexpected error during job recommendation.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while recommending jobs.")
