"""
ATS scoring service.

Design principles (see project instructions):
- Scores are TRANSPARENT: every component is returned individually, with its
  weight, so the frontend can show a real breakdown instead of a black-box number.
- Scores are NOT claimed to be equivalent to any commercial ATS system —
  see the `notes` field returned with every result.
- These are pure functions over pre-computed inputs (embeddings, extracted
  skill lists, raw text). They do not call the embedding model or skill
  extractor themselves — resume_service/matching_service (later modules)
  own that orchestration. This keeps scoring logic isolated and easy to
  unit test with fixed inputs, per the project's development rules.
- If a component cannot be meaningfully computed for a given input (e.g. a
  JD with no extractable skills), it is excluded and the remaining weights
  are renormalized to sum to 1.0, rather than silently scoring it as zero
  or as perfect.
"""

import re

import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS

from app.config import get_settings
from app.ml.similarity import cosine_similarity

settings = get_settings()

# Section headers we look for when checking resume completeness/structure.
# Matched case-insensitively as whole-word-ish substrings.
EXPECTED_SECTIONS = {
    "summary": ["summary", "objective", "profile"],
    "experience": ["experience", "employment", "work history"],
    "education": ["education", "academic"],
    "skills": ["skills", "technical skills", "competencies"],
    "projects": ["projects"],
    "certifications": ["certifications", "certificates", "licenses"],
}

# A resume showing this many or more distinct taxonomy skills is treated as
# having "full" skill breadth for the resume-only mode. Not a hard science —
# chosen as a reasonable point past which additional listed skills add
# diminishing signal about actual employability.
SKILL_BREADTH_FULL_SCORE_THRESHOLD = 15

ACTION_VERBS = {
    "developed", "built", "led", "designed", "implemented", "managed",
    "created", "improved", "increased", "reduced", "optimized", "automated",
    "deployed", "architected", "launched", "delivered", "analyzed",
    "collaborated", "mentored", "trained", "researched", "presented",
    "coordinated", "streamlined", "engineered", "established", "achieved",
}

QUANTIFIED_METRIC_PATTERN = re.compile(r"\d+(\.\d+)?\s*%|\$\s?\d+|\b\d{2,}\b")


# ---------------------------------------------------------------------------
# Shared helpers (used by both resume-only and resume+JD scoring)
# ---------------------------------------------------------------------------

def _resume_completeness_score(resume_text: str) -> float:
    """
    Fraction of expected resume sections that appear to be present,
    detected via simple case-insensitive header keyword search.
    Returns a 0-100 score.
    """
    text_lower = resume_text.lower()
    sections_found = 0

    for _section, keywords in EXPECTED_SECTIONS.items():
        if any(keyword in text_lower for keyword in keywords):
            sections_found += 1

    return (sections_found / len(EXPECTED_SECTIONS)) * 100


def _resume_structure_score(resume_text: str) -> float:
    """
    Deterministic proxy for resume writing quality: checks for action-verb-led
    bullet lines and quantified achievements (numbers, percentages, currency).
    This is intentionally simple — it is a fallback signal used when Groq is
    unavailable, and a component even when Groq IS available, since the
    project must never rely on the LLM as the only source of this signal.
    """
    lines = [line.strip() for line in resume_text.split("\n") if line.strip()]
    if not lines:
        return 0.0

    action_verb_lines = 0
    quantified_lines = 0

    for line in lines:
        first_word = re.sub(r"^[\-\*\u2022\s]+", "", line).split(" ")[0].lower().strip(".,:;")
        if first_word in ACTION_VERBS:
            action_verb_lines += 1
        if QUANTIFIED_METRIC_PATTERN.search(line):
            quantified_lines += 1

    action_verb_ratio = action_verb_lines / len(lines)
    quantified_ratio = quantified_lines / len(lines)

    # Weighted: action-verb usage matters somewhat more than quantification,
    # since not every genuine achievement has a natural number attached.
    score = (action_verb_ratio * 0.6 + quantified_ratio * 0.4) * 100

    # Ratios here are naturally small (most resume lines aren't bullet points),
    # so rescale against a realistic "good resume" ceiling rather than 100%
    # of ALL lines needing to qualify.
    REALISTIC_CEILING = 0.35
    scaled = min(100.0, (score / (REALISTIC_CEILING * 100)) * 100)
    return scaled


def _keyword_relevance_score(resume_text: str, jd_text: str, top_k: int = 30) -> float | None:
    """
    Deterministic keyword overlap: takes the top_k most frequent meaningful
    (non-stopword, length >= 3) words from the JD and checks what fraction
    also appear in the resume. Independent of the curated skills taxonomy —
    catches domain terms, tools, or phrasing not in that list.
    Returns None if the JD has no meaningful keywords to compare (too short).
    """
    def meaningful_words(text: str) -> list[str]:
        words = re.findall(r"[a-zA-Z][a-zA-Z\-\+#\.]{2,}", text.lower())
        return [w for w in words if w not in STOP_WORDS]

    jd_words = meaningful_words(jd_text)
    if not jd_words:
        return None

    # Frequency-rank JD words, take the top_k most common as "important" keywords.
    freq: dict[str, int] = {}
    for w in jd_words:
        freq[w] = freq.get(w, 0) + 1
    top_keywords = sorted(freq, key=freq.get, reverse=True)[:top_k]

    resume_words = set(meaningful_words(resume_text))
    found = sum(1 for kw in top_keywords if kw in resume_words)

    return (found / len(top_keywords)) * 100 if top_keywords else None


_YEARS_REQUIRED_PATTERN = re.compile(
    r"(\d+)\+?\s*(?:-|to)?\s*(\d+)?\s*\+?\s*years?", re.IGNORECASE
)
_YEAR_MENTION_PATTERN = re.compile(r"\b(19[8-9]\d|20[0-4]\d)\b")


def _experience_alignment_score(resume_text: str, jd_text: str) -> float | None:
    """
    Heuristic experience match. This is intentionally approximate — it does
    NOT parse structured work history or verify dates precisely. It:
      1. Looks for a stated years-of-experience requirement in the JD
         (e.g. "3+ years", "5-7 years").
      2. Estimates the candidate's experience span from the range of years
         mentioned in the resume (earliest to latest, treating "present"/
         "current" as the current year).
      3. Scores how well the estimated experience meets the requirement.

    Returns None if the JD does not state a years-of-experience requirement
    at all — in that case there is nothing meaningful to align against,
    and the component is excluded rather than guessed at.
    """
    jd_match = _YEARS_REQUIRED_PATTERN.search(jd_text)
    if not jd_match:
        return None

    low = int(jd_match.group(1))
    high = int(jd_match.group(2)) if jd_match.group(2) else low
    required_years = (low + high) / 2

    resume_lower = resume_text.lower()
    years_mentioned = [int(y) for y in _YEAR_MENTION_PATTERN.findall(resume_text)]

    has_present = "present" in resume_lower or "current" in resume_lower
    if has_present:
        import datetime
        years_mentioned.append(datetime.datetime.now().year)

    if len(years_mentioned) < 2:
        # Not enough date information to estimate a span at all.
        return None

    estimated_years = max(years_mentioned) - min(years_mentioned)
    estimated_years = max(0, estimated_years)

    if required_years <= 0:
        return 100.0

    ratio = estimated_years / required_years
    score = min(100.0, ratio * 100)
    return score


def _skill_breadth_score(resume_skills: list[str]) -> float:
    """
    Resume-only mode: proxy for how much relevant technical breadth a resume
    demonstrates, based on count of distinct taxonomy skills detected.
    Not a quality judgment — a resume that pads itself with irrelevant
    skill keywords would score well here too, which is a known limitation
    of any keyword-based approach and worth stating plainly.
    """
    count = len(resume_skills)
    return min(100.0, (count / SKILL_BREADTH_FULL_SCORE_THRESHOLD) * 100)


def _weighted_average(components: list[tuple[str, float | None, float]]) -> tuple[float, dict]:
    """
    Combines (name, score, weight) tuples into a single weighted score,
    excluding any component whose score is None and renormalizing the
    remaining weights to sum to 1.0. Returns (overall_score, breakdown_dict).
    """
    usable = [(name, score, weight) for name, score, weight in components if score is not None]

    if not usable:
        raise ValueError("No scoreable components were provided.")

    total_weight = sum(weight for _, _, weight in usable)
    overall = sum(score * (weight / total_weight) for _, score, weight in usable)

    breakdown = {
        name: {
            "score": round(score, 1),
            "weight": round(weight / total_weight, 3),
        }
        for name, score, weight in usable
    }

    excluded = [name for name, score, _ in components if score is None]
    if excluded:
        breakdown["_excluded_components"] = excluded

    return round(overall, 1), breakdown


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------

def score_resume_only(
    resume_text: str,
    resume_skills: list[str],
    top_role_semantic_similarity: float,
) -> dict:
    """
    General resume-quality score, used when no JD is provided.

    Args:
        resume_text: raw extracted resume text.
        resume_skills: canonical skill names detected in the resume (Module 4).
        top_role_semantic_similarity: 0-100 semantic similarity of the resume
            to its single best-matching role profile (from role_matcher's
            top result) — this is how the embedding model contributes here.
    """
    completeness = _resume_completeness_score(resume_text)
    structure = _resume_structure_score(resume_text)
    skill_breadth = _skill_breadth_score(resume_skills)
    role_signal = max(0.0, min(100.0, top_role_semantic_similarity))

    overall, breakdown = _weighted_average([
        ("completeness", completeness, settings.weight_resume_completeness),
        ("skill_breadth", skill_breadth, settings.weight_resume_skills),
        ("structure_quality", structure, settings.weight_resume_structure),
        ("role_alignment_signal", role_signal, settings.weight_resume_role_signal),
    ])

    return {
        "overall_score": overall,
        "breakdown": breakdown,
        "notes": (
            "This is a general resume-quality indicator based on structure, "
            "detected skills, writing patterns, and similarity to common role "
            "profiles. It is not a measurement of how any specific company's "
            "ATS software would score this resume."
        ),
    }


def score_resume_jd_match(
    resume_text: str,
    jd_text: str,
    resume_embedding: np.ndarray,
    jd_embedding: np.ndarray,
    resume_skills: list[str],
    jd_skills: list[str],
) -> dict:
    """
    ATS match score for a specific resume-JD pair.

    Args:
        resume_text, jd_text: raw extracted text for each.
        resume_embedding, jd_embedding: 768-dim vectors from embedding_model.
        resume_skills, jd_skills: canonical skill names detected in each (Module 4).
    """
    semantic_raw = cosine_similarity(resume_embedding, jd_embedding)
    semantic_score = max(0.0, min(100.0, semantic_raw * 100))

    if jd_skills:
        overlap = len(set(resume_skills) & set(jd_skills))
        skills_match_score = (overlap / len(jd_skills)) * 100
    else:
        skills_match_score = None  # nothing to compare against

    keyword_score = _keyword_relevance_score(resume_text, jd_text)
    experience_score = _experience_alignment_score(resume_text, jd_text)
    completeness_score = _resume_completeness_score(resume_text)

    overall, breakdown = _weighted_average([
        ("semantic_similarity", semantic_score, settings.weight_semantic),
        ("skills_match", skills_match_score, settings.weight_skills),
        ("keyword_relevance", keyword_score, settings.weight_keywords),
        ("experience_alignment", experience_score, settings.weight_experience),
        ("resume_completeness", completeness_score, settings.weight_completeness),
    ])

    return {
        "overall_score": overall,
        "breakdown": breakdown,
        "notes": (
            "This score combines semantic similarity, detected skill overlap, "
            "keyword relevance, an approximate experience-alignment check, and "
            "resume structure — it is a transparent, explainable estimate, not "
            "an equivalent of any specific commercial ATS system's scoring."
        ),
    }
