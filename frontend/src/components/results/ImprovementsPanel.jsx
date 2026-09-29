import { useState } from "react";
import { Loader2, Lightbulb } from "lucide-react";
import { recommendImprovements, ApiError } from "../../api/client";

const TIERS = [
  {
    key: "critical_improvements",
    title: "Critical",
    description: "Important JD requirements not clearly represented in your resume.",
    badgeClass: "bg-critical-soft text-critical",
  },
  {
    key: "recommended_improvements",
    title: "Recommended",
    description: "Changes that would meaningfully strengthen this match.",
    badgeClass: "bg-amber-soft text-amber",
  },
  {
    key: "optional_improvements",
    title: "Optional",
    description: "Minor polish.",
    badgeClass: "bg-teal-soft text-teal",
  },
];

export default function ImprovementsPanel({ resumeText, jdText }) {
  const [improvements, setImprovements] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleShow() {
    setIsLoading(true);
    setError("");
    try {
      const data = await recommendImprovements(resumeText, jdText);
      setImprovements(data);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not load recommendations.");
    } finally {
      setIsLoading(false);
    }
  }

  if (improvements === null) {
    return (
      <div className="text-center">
        <button
          type="button"
          onClick={handleShow}
          disabled={isLoading}
          className="inline-flex items-center gap-2 rounded-full border border-teal px-5 py-2.5 text-sm font-medium text-teal transition-colors hover:bg-teal-soft disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isLoading ? <Loader2 className="animate-spin" size={16} /> : <Lightbulb size={16} />}
          {isLoading ? "Preparing suggestions..." : "Show Recommendations / Improvements"}
        </button>
        {error && <p className="mt-2 text-sm text-critical">{error}</p>}
      </div>
    );
  }

  return (
    <div className="animate-unfold space-y-5">
      <h3 className="font-display text-lg font-semibold text-ink">Recommendations / Improvements</h3>
      {TIERS.map(({ key, title, description, badgeClass }) => {
        const items = improvements[key] || [];
        return (
          <div key={key}>
            <div className="mb-1.5 flex items-center gap-2">
              <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${badgeClass}`}>
                {title}
              </span>
              <span className="text-xs text-ink-muted">{description}</span>
            </div>
            {items.length === 0 ? (
              <p className="text-sm text-ink-muted">Nothing here.</p>
            ) : (
              <ul className="space-y-1.5">
                {items.map((item, i) => (
                  <li key={i} className="text-sm text-ink">
                    • {item}
                  </li>
                ))}
              </ul>
            )}
          </div>
        );
      })}
    </div>
  );
}
