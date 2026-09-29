import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

vi.mock("../../api/client", () => ({
  recommendImprovements: vi.fn(),
  ApiError: class ApiError extends Error {},
}));

import MatchResults from "./MatchResults";
import { recommendImprovements } from "../../api/client";

const mockData = {
  ats_match_score: 64.2,
  score_breakdown: {
    semantic_similarity: { score: 70, weight: 0.353 },
    skills_match: { score: 60, weight: 0.353 },
    keyword_relevance: { score: 55, weight: 0.176 },
    resume_completeness: { score: 100, weight: 0.118 },
  },
  excluded_score_components: ["experience_alignment"],
  score_notes: "This score combines semantic similarity, skill overlap, and more.",
  matching_skills: ["Python", "FastAPI"],
  missing_skills: ["AWS", "Kubernetes"],
  strengths: ["Your resume includes the key sections expected by most ATS systems."],
  jd_job_title: "Machine Learning Engineer",
  jd_required_skills: ["Python", "AWS"],
  jd_preferred_skills: ["Kubernetes"],
  jd_analysis_source: "fallback",
  embedding_model_type: "pretrained_fallback",
  resume_text: "full resume text long enough to pass validation",
  jd_text: "full jd text long enough to pass validation",
};

describe("MatchResults", () => {
  it("renders the match score and JD title", () => {
    render(<MatchResults data={mockData} />);
    expect(screen.getByText("64")).toBeInTheDocument();
    expect(screen.getByText("Machine Learning Engineer")).toBeInTheDocument();
  });

  it("renders matching and missing skills in separate sections", () => {
    render(<MatchResults data={mockData} />);
    expect(screen.getByText("Python")).toBeInTheDocument();
    expect(screen.getByText("AWS")).toBeInTheDocument();
    expect(screen.getByText("Matching skills")).toBeInTheDocument();
    expect(screen.getByText("Missing skills")).toBeInTheDocument();
  });

  it("shows the excluded component note", () => {
    render(<MatchResults data={mockData} />);
    expect(screen.getByText(/experience alignment/i)).toBeInTheDocument();
  });

  it("fetches improvements only after the button is clicked, and separates tiers", async () => {
    recommendImprovements.mockResolvedValue({
      critical_improvements: ["Add AWS experience explicitly."],
      recommended_improvements: ["Mention Kubernetes if you have used it."],
      optional_improvements: ["Polish your summary wording."],
      source: "fallback",
    });

    render(<MatchResults data={mockData} />);

    expect(recommendImprovements).not.toHaveBeenCalled();
    expect(screen.queryByText(/add aws experience/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /show recommendations/i }));

    await waitFor(() =>
      expect(recommendImprovements).toHaveBeenCalledWith(mockData.resume_text, mockData.jd_text)
    );

    expect(await screen.findByText(/add aws experience/i)).toBeInTheDocument();
    expect(screen.getByText(/mention kubernetes/i)).toBeInTheDocument();
    expect(screen.getByText(/polish your summary/i)).toBeInTheDocument();
    expect(screen.getByText("Critical")).toBeInTheDocument();
    expect(screen.getByText("Recommended")).toBeInTheDocument();
    expect(screen.getByText("Optional")).toBeInTheDocument();
  });
});
