import { useState } from "react";
import Header from "./components/layout/Header";
import ResumeUploadCard from "./components/upload/ResumeUploadCard";
import JDInputCard from "./components/upload/JDInputCard";
import AnalyzeButton from "./components/upload/AnalyzeButton";
import { analyzeResume, analyzeMatch, ApiError } from "./api/client";
import ResumeResults from "./components/results/ResumeResults";
import MatchResults from "./components/results/MatchResults";

const MIN_JD_TEXT_LENGTH = 20;

function App() {
  const [resumeFile, setResumeFile] = useState(null);

  const [jdInputMethod, setJdInputMethod] = useState("text");
  const [jdText, setJdText] = useState("");
  const [jdFile, setJdFile] = useState(null);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null); // { mode: "resume-only" | "match", data }

  const jdProvided =
    (jdInputMethod === "text" && jdText.trim().length >= MIN_JD_TEXT_LENGTH) ||
    (jdInputMethod === "file" && jdFile !== null);

  const jdTextTooShort =
    jdInputMethod === "text" && jdText.trim().length > 0 && jdText.trim().length < MIN_JD_TEXT_LENGTH;

  const canAnalyze = resumeFile !== null && !jdTextTooShort;

  function handleJdInputMethodChange(method) {
    setJdInputMethod(method);
    // Switching method clears the other, so only one JD source is ever active.
    if (method === "text") setJdFile(null);
    if (method === "file") setJdText("");
  }

  async function handleAnalyze() {
    setIsAnalyzing(true);
    setError("");
    setResult(null);

    try {
      if (jdProvided) {
        const jdSource =
          jdInputMethod === "file" ? { type: "file", value: jdFile } : { type: "text", value: jdText };
        const data = await analyzeMatch(resumeFile, jdSource);
        setResult({ mode: "match", data });
      } else {
        const data = await analyzeResume(resumeFile);
        setResult({ mode: "resume-only", data });
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Something went wrong. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <div className="min-h-screen bg-paper">
      <Header />
      <main className="mx-auto max-w-4xl px-6 py-10">
        <div className="space-y-5">
          <ResumeUploadCard
            file={resumeFile}
            onFileSelect={(file) => {
              setResumeFile(file);
              setResult(null);
            }}
            onRemove={() => setResumeFile(null)}
          />

          <JDInputCard
            inputMethod={jdInputMethod}
            onInputMethodChange={handleJdInputMethodChange}
            jdText={jdText}
            onTextChange={setJdText}
            jdFile={jdFile}
            onFileSelect={setJdFile}
            onRemoveFile={() => setJdFile(null)}
          />

          <AnalyzeButton
            disabled={!canAnalyze}
            isLoading={isAnalyzing}
            onClick={handleAnalyze}
            helperText={
              !resumeFile
                ? "Upload a resume to get started."
                : jdProvided
                ? "We'll score your resume against this job description."
                : "We'll give a general resume review — add a job description above for a match score instead."
            }
          />

          {error && (
            <div className="rounded border border-critical/30 bg-critical-soft px-4 py-3 text-sm text-critical">
              {error}
            </div>
          )}

          {result && result.mode === "resume-only" && <ResumeResults data={result.data} />}
          {result && result.mode === "match" && <MatchResults data={result.data} />}
        </div>
      </main>
    </div>
  );
}

export default App;
