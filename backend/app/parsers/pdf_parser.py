"""
PDF text extraction.

Uses pdfplumber (free, open-source) rather than PyPDF2 because it handles
multi-column layouts and spacing more reliably for resume-style documents.

Known limitation: this does NOT do OCR. A scanned/image-only PDF (no embedded
text layer) will extract to an empty or near-empty string. That case is
detected and raised as a clear error rather than silently returning nothing.
"""

import io

import pdfplumber

from app.parsers.file_validator import FileValidationError

# Below this character count, we treat extraction as having effectively failed
# (e.g. a scanned PDF where only a page header/footer had real text).
MIN_MEANINGFUL_CHARS = 40


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts and returns all text from a PDF's pages, joined with newlines.
    Raises FileValidationError with a user-facing message on failure.
    """
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if len(pdf.pages) == 0:
                raise FileValidationError("The PDF file contains no pages.")

            page_texts = []
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    page_texts.append(extracted)

    except FileValidationError:
        raise
    except Exception as exc:
        # pdfplumber raises various underlying exceptions (PDFSyntaxError, etc.)
        # for corrupted or malformed files. Normalize all of them to one clear message.
        raise FileValidationError(
            "Could not read the PDF file. It may be corrupted or password-protected."
        ) from exc

    full_text = "\n".join(page_texts).strip()

    if len(full_text) < MIN_MEANINGFUL_CHARS:
        raise FileValidationError(
            "No readable text could be extracted from this PDF. "
            "It may be a scanned image without a text layer. "
            "Please upload a text-based PDF or a DOCX file instead."
        )

    return full_text
