"""
Single entry point for text extraction, dispatching to the correct parser
based on file extension. Services (resume_service, matching_service) call
this rather than importing pdf_parser / docx_parser directly.
"""

from app.parsers.docx_parser import extract_text_from_docx
from app.parsers.file_validator import FileValidationError
from app.parsers.pdf_parser import extract_text_from_pdf


def extract_text(file_bytes: bytes, extension: str) -> str:
    """
    Routes to the appropriate parser based on extension (".pdf" or ".docx").
    Assumes extension has already been validated by file_validator.validate_extension.
    """
    if extension == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif extension == ".docx":
        return extract_text_from_docx(file_bytes)
    else:
        # Defensive fallback — should be unreachable if validate_extension ran first.
        raise FileValidationError(f"Unsupported file extension: {extension}")
