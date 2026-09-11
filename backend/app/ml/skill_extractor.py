"""
Deterministic skill extraction.

Matches resume/JD text against a curated skills taxonomy (app/data/skills_taxonomy.json)
using spaCy's PhraseMatcher. This is intentionally NOT LLM-based: skill extraction
needs to be consistent and auditable, since it feeds directly into ATS scoring
(skills_match component) and into missing-skills / matching-skills comparisons.

Uses spacy.blank("en") — a tokenizer only, no pretrained pipeline/model download
required. PhraseMatcher just needs a vocab, not word vectors or a POS tagger.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

import spacy
from spacy.matcher import PhraseMatcher

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "skills_taxonomy.json"


class SkillExtractor:
    """
    Loads the skills taxonomy once and builds a PhraseMatcher over all
    skill aliases. `extract` can then be called repeatedly and cheaply.
    """

    def __init__(self, taxonomy_path: Path = _DATA_PATH) -> None:
        self.nlp = spacy.blank("en")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        # alias_doc_id -> canonical skill info, so a match can be mapped back
        # to its canonical name/category regardless of which alias matched.
        self._match_id_to_skill: dict[str, dict] = {}

        self._build(taxonomy_path)

    def _build(self, taxonomy_path: Path) -> None:
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)

        for skill in taxonomy["skills"]:
            canonical = skill["canonical"]
            category = skill["category"]
            aliases = skill["aliases"]

            # Use the canonical name as the match label so `matcher(doc)` results
            # can be mapped straight back to skill info via self.nlp.vocab.strings.
            match_id = canonical
            self._match_id_to_skill[match_id] = {"canonical": canonical, "category": category}

            patterns = [self.nlp.make_doc(alias) for alias in aliases]
            self.matcher.add(match_id, patterns)

        logger.info(f"Skill extractor loaded {len(taxonomy['skills'])} skills.")

    def extract(self, text: str) -> list[dict]:
        """
        Returns a deduplicated list of skills found in `text`, each as
        {"canonical": ..., "category": ...}, sorted by category then name.
        Returns an empty list for empty/whitespace-only input rather than raising —
        callers may legitimately pass partial/short text (e.g. a short JD snippet).
        """
        if not text or not text.strip():
            return []

        doc = self.nlp.make_doc(text)
        matches = self.matcher(doc)

        found_canonicals: set[str] = set()
        results: list[dict] = []

        for match_id, _start, _end in matches:
            label = self.nlp.vocab.strings[match_id]
            skill_info = self._match_id_to_skill[label]
            if skill_info["canonical"] not in found_canonicals:
                found_canonicals.add(skill_info["canonical"])
                results.append(skill_info)

        results.sort(key=lambda s: (s["category"], s["canonical"]))
        return results

    def extract_names(self, text: str) -> list[str]:
        """Convenience wrapper returning just canonical skill names, no category."""
        return [s["canonical"] for s in self.extract(text)]


@lru_cache
def get_skill_extractor() -> SkillExtractor:
    """
    Cached accessor — builds the PhraseMatcher once per process, not once
    per request. Import this function elsewhere rather than instantiating
    SkillExtractor directly.
    """
    return SkillExtractor()
