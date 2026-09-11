"""
File validation utilities.

These checks run BEFORE any parsing is attempted, so bad input fails fast
with a clear error instead of crashing deep inside a parser.
"""

from fastapi import UploadFile

from app.config import get_settings

settings = get_settings()

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


class FileValidationError(Exception):
    """Raised when an uploaded file fails validation. Caught in the route layer
    and converted into a proper HTTP error response."""


def validate_extension(filename: str) -> str:
    """
    Confirms the filename has an allowed extension.
    Returns the lowercase extension (e.g. ".pdf") for downstream use.
    """
    if not filename or "." not in filename:
        raise FileValidationError(
            "Uploaded file has no extension. Only PDF and DOCX are supported."
        )

    extension = "." + filename.rsplit(".", 1)[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported file type '{extension}'. Only PDF and DOCX are supported."
        )

    return extension


def validate_file_size(file_bytes: bytes) -> None:
    """
    Confirms the file does not exceed the configured size limit.
    Also rejects genuinely empty uploads (0 bytes) early.
    """
    if len(file_bytes) == 0:
        raise FileValidationError("Uploaded file is empty.")

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise FileValidationError(
            f"File exceeds the maximum allowed size of {settings.max_file_size_mb} MB."
        )


async def read_and_validate_upload(upload_file: UploadFile) -> tuple[bytes, str]:
    """
    Reads an UploadFile fully into memory, validates extension and size,
    and returns (raw_bytes, extension).

    Kept as a single entry point so every route that accepts a file
    (resume upload, JD upload) applies identical checks.
    """
    extension = validate_extension(upload_file.filename)
    file_bytes = await upload_file.read()
    validate_file_size(file_bytes)
    return file_bytes, extension
