import re
import pandas as pd
from nltk.tokenize import sent_tokenize


def clean_text(text):
    """
    Cleans raw text for NLP processing.

    Steps:
    - Convert to lowercase
    - Remove URLs
    - Remove email addresses
    - Remove extra whitespace
    - Remove unwanted symbols
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", " ", text)

    # remove emails
    text = re.sub(r"\S+@\S+", " ", text)

    # remove unwanted characters
    text = re.sub(r"[^a-zA-Z0-9+#./ ]", " ", text)

    # remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_job_title(title):

    title = str(title).lower()

    # Remove punctuation
    title = re.sub(r"[^a-zA-Z0-9\s/+]", " ", title)

    # Remove multiple spaces
    title = re.sub(r"\s+", " ", title)

    return title.strip()


def normalize_job_title(title):

    title = title.lower()

    title = re.sub(r"\bsr\b|\bsenior\b", "", title)
    title = re.sub(r"\bjr\b|\bjunior\b", "", title)
    title = re.sub(r"\blead\b", "", title)
    title = re.sub(r"\bprincipal\b", "", title)
    title = re.sub(r"\bassociate\b", "", title)

    title = re.sub(r"\s+", " ", title)

    return title.strip()


def create_chunks(text, chunk_size=5, overlap=2):
    """
    Create overlapping chunks from sentences.

    Parameters
    ----------
    chunk_size : number of sentences per chunk
    overlap : number of overlapping sentences
    """

    if pd.isna(text):
        return []

    sentences = sent_tokenize(text)
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + chunk_size])
        if len(chunk.strip()) > 0:
            chunks.append(chunk)
    return chunks


BOILERPLATE_PHRASES = [
    "hereby declare",
    "hereby confirm",
    "i declare",
    "declaration",
    "references available upon request",
    "curriculum vitae",
    "date place signature",
    "certify that",
    "to the best of my knowledge",
]

CONTACT_KEYWORDS = [
    "email phone",
    "phone email",
    "linkedin",
    "github.com",
    "portfolio",
]


def is_boilerplate_chunk(text, min_words=25):
    """
    Flags chunks that carry no real skill/experience signal:
    contact/header blocks and legal declaration boilerplate.

    A chunk is flagged if:
    - it contains a known declaration phrase, OR
    - it is short (<= min_words) AND contains either a long digit run
      (phone number) or a contact-related keyword (email/linkedin/etc.)

    Note: clean_text() already strips actual email addresses and URLs,
    so detection relies on the words/digits left behind (e.g. "email",
    "phone", a 7+ digit phone number, "linkedin").
    """

    if not text or not text.strip():
        return True

    lowered = text.lower()
    word_count = len(text.split())

    for phrase in BOILERPLATE_PHRASES:
        if phrase in lowered:
            return True

    has_long_digit_run = re.search(r"\d{7,}", lowered) is not None
    has_contact_keyword = any(kw in lowered for kw in CONTACT_KEYWORDS)

    if word_count <= min_words and (has_long_digit_run or has_contact_keyword):
        return True

    return False


def create_chunk_objects(chunk_list):
    """
    Converts a list of text chunks into a list of dictionaries.

    Example:
    [
        {"chunk_id":0, "text":"..."},
        {"chunk_id":1, "text":"..."}
    ]
    """

    return [
        {
            "chunk_id": idx,
            "text": chunk
        }
        for idx, chunk in enumerate(chunk_list)
    ]
