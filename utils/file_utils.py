import io
from pypdf import PdfReader
import docx


def extract_text_from_file(uploaded_file):
    """
    Extracts raw text from an uploaded .pdf, .docx, or .txt file.
    uploaded_file : a Streamlit UploadedFile object
    """

    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
        return text

    elif filename.endswith(".docx"):
        document = docx.Document(io.BytesIO(uploaded_file.getvalue()))
        text = "\n".join(p.text for p in document.paragraphs)
        return text

    elif filename.endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")

    else:
        raise ValueError("Unsupported file type. Please upload a .pdf, .docx, or .txt file.")
