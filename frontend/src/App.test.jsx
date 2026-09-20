import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

vi.mock("./api/client", () => ({
  analyzeResume: vi.fn(),
  analyzeMatch: vi.fn(),
  ApiError: class ApiError extends Error {},
}));

import App from "./App";
import { analyzeResume, analyzeMatch } from "./api/client";

function makeFile(name, type = "application/pdf") {
  return new File(["dummy content long enough to pass size checks"], name, { type });
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("Upload flow", () => {
  it("disables Analyze until a resume is selected", () => {
    render(<App />);
    const button = screen.getByRole("button", { name: /analyze resume/i });
    expect(button).toBeDisabled();
  });

  it("enables Analyze once a resume file is selected", async () => {
    render(<App />);
    const fileInputs = document.querySelectorAll('input[type="file"]');
    const resumeInput = fileInputs[0]; // Resume card renders first

    fireEvent.change(resumeInput, { target: { files: [makeFile("resume.pdf")] } });

    const button = screen.getByRole("button", { name: /analyze resume/i });
    await waitFor(() => expect(button).not.toBeDisabled());
  });

  it("calls analyzeResume (not analyzeMatch) when no JD is provided", async () => {
    analyzeResume.mockResolvedValue({
      ats_score: 72,
      score_breakdown: { completeness: { score: 80, weight: 1 } },
      excluded_score_components: [],
      score_notes: "notes",
      summary: "summary",
      skills_by_category: {},
      strengths: [],
      areas_for_improvement: [],
      resume_text: "resume text",
    });
    render(<App />);

    const fileInputs = document.querySelectorAll('input[type="file"]');
    fireEvent.change(fileInputs[0], { target: { files: [makeFile("resume.pdf")] } });

    const button = await screen.findByRole("button", { name: /analyze resume/i });
    await waitFor(() => expect(button).not.toBeDisabled());
    fireEvent.click(button);

    await waitFor(() => expect(analyzeResume).toHaveBeenCalledTimes(1));
    expect(analyzeMatch).not.toHaveBeenCalled();
  });

  it("calls analyzeMatch when JD text is provided alongside a resume", async () => {
    analyzeMatch.mockResolvedValue({
      ats_match_score: 85,
      score_breakdown: { semantic_similarity: { score: 85, weight: 1 } },
      excluded_score_components: [],
      score_notes: "notes",
      matching_skills: ["Python"],
      missing_skills: ["AWS"],
      strengths: [],
      jd_job_title: "Python Developer",
      jd_required_skills: [],
      jd_preferred_skills: [],
      resume_text: "resume text",
      jd_text: "jd text",
    });
    const user = userEvent.setup();
    render(<App />);

    const fileInputs = document.querySelectorAll('input[type="file"]');
    fireEvent.change(fileInputs[0], { target: { files: [makeFile("resume.pdf")] } });

    const textarea = screen.getByPlaceholderText(/paste the job description/i);
    await user.type(textarea, "We need a Python developer with 3+ years of experience.");

    const button = await screen.findByRole("button", { name: /analyze resume/i });
    await waitFor(() => expect(button).not.toBeDisabled());
    fireEvent.click(button);

    await waitFor(() => expect(analyzeMatch).toHaveBeenCalledTimes(1));
    expect(analyzeResume).not.toHaveBeenCalled();
  });

  it("disables Analyze when JD text is non-empty but below the minimum length", async () => {
    render(<App />);
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fireEvent.change(fileInputs[0], { target: { files: [makeFile("resume.pdf")] } });

    const textarea = screen.getByPlaceholderText(/paste the job description/i);
    fireEvent.change(textarea, { target: { value: "too short" } });

    const button = screen.getByRole("button", { name: /analyze resume/i });
    await waitFor(() => expect(button).toBeDisabled());
    expect(screen.getByText(/characters minimum/i)).toBeInTheDocument();
  });

  it("switching JD input method clears the other input's value", async () => {
    render(<App />);
    const textarea = screen.getByPlaceholderText(/paste the job description/i);
    fireEvent.change(textarea, { target: { value: "some job description text here" } });

    fireEvent.click(screen.getByRole("button", { name: /upload file/i }));
    fireEvent.click(screen.getByRole("button", { name: /^paste text$/i }));

    const textareaAgain = screen.getByPlaceholderText(/paste the job description/i);
    expect(textareaAgain.value).toBe("");
  });

  it("shows an error message when the API call fails", async () => {
    analyzeResume.mockRejectedValue(Object.assign(new Error("fail"), { detail: "Backend exploded" }));
    render(<App />);

    const fileInputs = document.querySelectorAll('input[type="file"]');
    fireEvent.change(fileInputs[0], { target: { files: [makeFile("resume.pdf")] } });

    const button = await screen.findByRole("button", { name: /analyze resume/i });
    await waitFor(() => expect(button).not.toBeDisabled());
    fireEvent.click(button);

    await waitFor(() => expect(screen.getByText(/backend exploded|something went wrong/i)).toBeInTheDocument());
  });
});
