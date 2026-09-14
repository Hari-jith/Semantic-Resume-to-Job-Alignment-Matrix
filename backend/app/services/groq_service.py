"""
Groq LLM integration.

Used ONLY for language tasks: summarizing, explaining, and phrasing
recommendations. It is never the sole source of scoring or of skill/JD
extraction used for scoring — see ats_service.py and skill_extractor.py for
the deterministic components those rely on.

Every Groq call:
  1. Asks for JSON-only output via response_format={"type": "json_object"}.
  2. Validates the parsed JSON against a strict internal Pydantic model.
  3. Falls back to a deterministic, clearly-labeled result if the call fails,
     times out, or returns something that doesn't validate — so the app
     keeps working end-to-end even with GROQ_API_KEY unset or Groq down.

Every function returns a dict that includes "source": "groq" | "fallback"
so callers (and the frontend) always know which path produced a given result.
"""

import json
import logging

from groq import Groq
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Internal schemas — validate Groq's raw JSON output before it is trusted.
# (Separate from the API-facing request/response schemas added in later modules.)
# ---------------------------------------------------------------------------

class ResumeAnalysisLLMOutput(BaseModel):
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience_summary: str = ""
    education_summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)


class JDAnalysisLLMOutput(BaseModel):
    job_title: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    experience_requirements: str = ""


class ImprovementsLLMOutput(BaseModel):
    critical_improvements: list[str] = Field(default_factory=list)
    recommended_improvements: list[str] = Field(default_factory=list)
    optional_improvements: list[str] = Field(default_factory=list)


class RoleExplanationsLLMOutput(BaseModel):
    # role_name -> one or two sentence explanation
    explanations: dict[str, str] = Field(default_factory=dict)


class GroqUnavailableError(Exception):
    """Raised internally when Groq cannot be used (no key, or call failed after retries)."""


# ---------------------------------------------------------------------------
# Client + low-level call wrapper
# ---------------------------------------------------------------------------

class GroqService:
    def __init__(self) -> None:
        self._client: Groq | None = None
        if settings.groq_api_key:
            self._client = Groq(api_key=settings.groq_api_key)
        else:
            logger.warning(
                "GROQ_API_KEY is not set. LLM-powered analysis will use "
                "deterministic fallbacks for all requests."
            )

    def is_available(self) -> bool:
        return self._client is not None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Low-level call: sends a system+user prompt, requests JSON-only
        output, and returns the parsed dict. Retries transient failures
        up to 3 times with exponential backoff. Raises on final failure —
        callers are responsible for catching and falling back.
        """
        if self._client is None:
            raise GroqUnavailableError("Groq client not configured (no API key).")

        response = self._client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_completion_tokens=1500,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        return json.loads(content)

    # -----------------------------------------------------------------
    # Resume analysis
    # -----------------------------------------------------------------

    def analyze_resume(self, resume_text: str, deterministic_skills: list[str]) -> dict:
        """
        Returns a structured resume analysis. Falls back to a rule-based
        summary built from `deterministic_skills` (Module 4 extraction) and
        simple heuristics if Groq is unavailable or its output doesn't validate.
        """
        system_prompt = (
            "You are an assistant that analyzes resumes for an ATS tool. "
            "Respond with ONLY a JSON object, no other text, matching exactly this shape: "
            '{"summary": string, "skills": [string], "experience_summary": string, '
            '"education_summary": string, "strengths": [string], "improvements": [string]}. '
            "Base every field strictly on the resume text provided. Do not invent "
            "companies, dates, or skills that are not present in the text. "
            "'improvements' must be honest, actionable suggestions — never suggest "
            "the candidate claim skills or experience they do not have."
        )
        user_prompt = f"Resume text:\n\n{resume_text}"

        if not self.is_available():
            return self._fallback_resume_analysis(resume_text, deterministic_skills)

        try:
            raw = self._call(system_prompt, user_prompt)
            validated = ResumeAnalysisLLMOutput.model_validate(raw)
            result = validated.model_dump()
            result["source"] = "groq"
            return result
        except (GroqUnavailableError, ValidationError, json.JSONDecodeError, Exception) as exc:
            logger.warning(f"Groq resume analysis unavailable, using fallback: {exc}")
            return self._fallback_resume_analysis(resume_text, deterministic_skills)

    @staticmethod
    def _fallback_resume_analysis(resume_text: str, deterministic_skills: list[str]) -> dict:
        first_lines = [l.strip() for l in resume_text.split("\n") if l.strip()][:3]
        summary = " ".join(first_lines)[:300] if first_lines else ""

        return {
            "summary": summary,
            "skills": deterministic_skills,
            "experience_summary": "",
            "education_summary": "",
            "strengths": [],
            "improvements": [
                "AI-generated analysis is temporarily unavailable. Skills shown "
                "were detected using keyword matching only."
            ],
            "source": "fallback",
        }

    # -----------------------------------------------------------------
    # Job description analysis
    # -----------------------------------------------------------------

    def analyze_job_description(self, jd_text: str, deterministic_skills: list[str]) -> dict:
        system_prompt = (
            "You are an assistant that analyzes job descriptions for an ATS tool. "
            "Respond with ONLY a JSON object, no other text, matching exactly this shape: "
            '{"job_title": string, "required_skills": [string], "preferred_skills": [string], '
            '"responsibilities": [string], "experience_requirements": string}. '
            "Base every field strictly on the job description text provided."
        )
        user_prompt = f"Job description text:\n\n{jd_text}"

        if not self.is_available():
            return self._fallback_jd_analysis(deterministic_skills)

        try:
            raw = self._call(system_prompt, user_prompt)
            validated = JDAnalysisLLMOutput.model_validate(raw)
            result = validated.model_dump()
            result["source"] = "groq"
            return result
        except (GroqUnavailableError, ValidationError, json.JSONDecodeError, Exception) as exc:
            logger.warning(f"Groq JD analysis unavailable, using fallback: {exc}")
            return self._fallback_jd_analysis(deterministic_skills)

    @staticmethod
    def _fallback_jd_analysis(deterministic_skills: list[str]) -> dict:
        return {
            "job_title": "",
            "required_skills": deterministic_skills,
            "preferred_skills": [],
            "responsibilities": [],
            "experience_requirements": "",
            "source": "fallback",
        }

    # -----------------------------------------------------------------
    # JD-specific improvement recommendations
    # -----------------------------------------------------------------

    def generate_improvements(
        self,
        resume_text: str,
        jd_text: str,
        matching_skills: list[str],
        missing_skills: list[str],
    ) -> dict:
        """
        Generates tiered, JD-specific improvement suggestions. The LLM is
        given the already-computed matching/missing skill lists so its
        suggestions are grounded in real gaps, not invented ones.
        """
        system_prompt = (
            "You are a career coach assistant helping a candidate improve their "
            "resume for a specific job description. Respond with ONLY a JSON object, "
            "no other text, matching exactly this shape: "
            '{"critical_improvements": [string], "recommended_improvements": [string], '
            '"optional_improvements": [string]}. '
            "Critical = important JD requirements clearly missing from the resume. "
            "Recommended = improvements that would meaningfully strengthen the match. "
            "Optional = minor polish. "
            "NEVER suggest the candidate claim skills, experience, or achievements "
            "they do not have. Only suggest honest changes: highlighting existing "
            "relevant experience, emphasizing real projects, improving wording, or "
            "adding skills the candidate already has but didn't list."
        )
        user_prompt = (
            f"Resume text:\n{resume_text}\n\n"
            f"Job description text:\n{jd_text}\n\n"
            f"Skills already matching between resume and JD: {matching_skills}\n"
            f"Skills required by the JD but not found in the resume: {missing_skills}"
        )

        if not self.is_available():
            return self._fallback_improvements(missing_skills)

        try:
            raw = self._call(system_prompt, user_prompt)
            validated = ImprovementsLLMOutput.model_validate(raw)
            result = validated.model_dump()
            result["source"] = "groq"
            return result
        except (GroqUnavailableError, ValidationError, json.JSONDecodeError, Exception) as exc:
            logger.warning(f"Groq improvements unavailable, using fallback: {exc}")
            return self._fallback_improvements(missing_skills)

    @staticmethod
    def _fallback_improvements(missing_skills: list[str]) -> dict:
        critical = [
            f"The job description emphasizes '{skill}', which was not found in your resume. "
            f"If you have experience with it, add it explicitly."
            for skill in missing_skills[:5]
        ]
        recommended = [
            f"Consider highlighting any experience with '{skill}' if applicable."
            for skill in missing_skills[5:10]
        ]
        return {
            "critical_improvements": critical,
            "recommended_improvements": recommended,
            "optional_improvements": [
                "AI-generated wording suggestions are temporarily unavailable."
            ],
            "source": "fallback",
        }

    # -----------------------------------------------------------------
    # Role match explanations (Resume-only mode, "Recommended Jobs")
    # -----------------------------------------------------------------

    def generate_role_explanations(self, ranked_roles: list[dict]) -> dict:
        """
        Given role_matcher's ranked output (each with role_name and
        matching_skills), asks Groq for a short natural-language explanation
        per role, grounded in the actual matching_skills — not invented.
        Falls back to a simple templated sentence per role if unavailable.

        Returns {"explanations": {role_name: explanation, ...}, "source": "groq"|"fallback"}
        — kept consistent with the other three methods' return shape.
        """
        system_prompt = (
            "You write short, specific one-to-two sentence explanations of why "
            "a candidate's resume matches a given job role, based ONLY on the "
            "overlapping skills provided. Respond with ONLY a JSON object of the "
            'exact shape {"explanations": {"<role name>": "<explanation>", ...}} '
            "with one entry per role given. Do not mention skills that are not "
            "in the provided overlap list for that role."
        )
        roles_summary = [
            {"role_name": r["role_name"], "matching_skills": r["matching_skills"]}
            for r in ranked_roles
        ]
        user_prompt = f"Roles and their overlapping skills:\n{json.dumps(roles_summary)}"

        if not self.is_available():
            return {"explanations": self._fallback_role_explanations(ranked_roles), "source": "fallback"}

        try:
            raw = self._call(system_prompt, user_prompt)
            validated = RoleExplanationsLLMOutput.model_validate(raw)
            return {"explanations": validated.explanations, "source": "groq"}
        except (GroqUnavailableError, ValidationError, json.JSONDecodeError, Exception) as exc:
            logger.warning(f"Groq role explanations unavailable, using fallback: {exc}")
            return {"explanations": self._fallback_role_explanations(ranked_roles), "source": "fallback"}

    @staticmethod
    def _fallback_role_explanations(ranked_roles: list[dict]) -> dict[str, str]:
        explanations = {}
        for role in ranked_roles:
            skills = role["matching_skills"]
            if skills:
                shown = ", ".join(skills[:4])
                explanations[role["role_name"]] = (
                    f"Your resume shows overlapping skills with this role, including {shown}."
                )
            else:
                explanations[role["role_name"]] = (
                    "This role was suggested based on overall resume similarity, "
                    "though no specific overlapping skills were detected."
                )
        return explanations


_groq_service: GroqService | None = None


def get_groq_service() -> GroqService:
    """Cached accessor — the Groq client is constructed once per process."""
    global _groq_service
    if _groq_service is None:
        _groq_service = GroqService()
    return _groq_service
