import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

vi.mock("../../api/client", () => ({
  recommendJobs: vi.fn(),
  ApiError: class ApiError extends Error {},
}));

import ResumeResults from "./ResumeResults";
import { recommendJobs } from "../../api/client";

const mockData = {
  ats_score: 78.5,
  score_breakdown: {
    completeness: { score: 100, weight: 0.35 },
    skill_breadth: { score: 60, weight: 0.3 },
    structure_quality: { score: 45, weight: 0.2 },
    role_alignment_signal: { score: 82, weight: 0.15 },
  },
  excluded_score_components: [],
  score_notes: "This is a general resume-quality indicator.",
  summary: "Experienced AI/ML engineer with a focus on NLP.",
  detected_skills: ["Python", "PyTorch"],
  skills_by_category: {
    "Programming Languages": ["Python"],
    "ML/DL Frameworks": ["PyTorch", "TensorFlow"],
  },
  strengths: ["Clear skills section", "Relevant project experience"],
  areas_for_improvement: ["Add quantified achievements"],
  analysis_source: "fallback",
  embedding_model_type: "pretrained_fallback",
  resume_text: "full resume text here, long enough to pass validation checks",
};

describe("ResumeResults", () => {
  it("renders the overall score, rounded", () => {
    render(<ResumeResults data={mockData} />);
    expect(screen.getByText("79")).toBeInTheDocument(); // 78.5 rounds to 79
  });

  it("renders the resume summary", () => {
    render(<ResumeResults data={mockData} />);
    expect(screen.getByText(/experienced ai\/ml engineer/i)).toBeInTheDocument();
  });

  it("renders score breakdown components with formatted labels", () => {
    render(<ResumeResults data={mockData} />);
    expect(screen.getByText("Skill Breadth")).toBeInTheDocument();
    expect(screen.getByText("Role Alignment Signal")).toBeInTheDocument();
  });

  it("renders skills grouped by category", () => {
    render(<ResumeResults data={mockData} />);
    expect(screen.getByText("Programming Languages")).toBeInTheDocument();
    expect(screen.getByText("PyTorch")).toBeInTheDocument();
  });

  it("renders strengths and areas for improvement", () => {
    render(<ResumeResults data={mockData} />);
    expect(screen.getByText("Clear skills section")).toBeInTheDocument();
    expect(screen.getByText("Add quantified achievements")).toBeInTheDocument();
  });

  it("fetches and displays recommended jobs only after the button is clicked", async () => {
    recommendJobs.mockResolvedValue({
      recommended_roles: [
        {
          role_name: "AI/ML Engineer",
          match_score: 88,
          semantic_similarity: 80,
          skill_match_score: 90,
          matching_skills: ["Python", "PyTorch"],
          explanation: "Strong overlap in ML tooling.",
        },
      ],
    });

    render(<ResumeResults data={mockData} />);

    // Not fetched yet
    expect(recommendJobs).not.toHaveBeenCalled();
    expect(screen.queryByText("AI/ML Engineer")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /show recommended jobs/i }));

    await waitFor(() => expect(recommendJobs).toHaveBeenCalledWith(mockData.resume_text));
    expect(await screen.findByText("AI/ML Engineer")).toBeInTheDocument();
    expect(screen.getByText(/88% match/i)).toBeInTheDocument();
    expect(screen.getByText(/strong overlap in ml tooling/i)).toBeInTheDocument();
  });

  it("shows excluded components note when present", () => {
    const dataWithExclusion = { ...mockData, excluded_score_components: ["experience_alignment"] };
    render(<ResumeResults data={dataWithExclusion} />);
    expect(screen.getByText(/experience alignment/i)).toBeInTheDocument();
  });
});
