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
