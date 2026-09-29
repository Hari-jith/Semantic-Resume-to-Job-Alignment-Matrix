import { useState } from "react";
import { Loader2, Briefcase } from "lucide-react";
import { recommendJobs, ApiError } from "../../api/client";

function RoleCard({ role }) {
  return (
    <div className="rounded border border-line bg-surface p-4">
      <div className="flex items-start justify-between gap-3">
        <h4 className="font-display text-base font-semibold text-ink">{role.role_name}</h4>
        <span className="shrink-0 rounded-full bg-teal-soft px-2.5 py-0.5 text-sm font-medium text-teal">
          {role.match_score.toFixed(0)}% match
        </span>
      </div>
      <p className="mt-1.5 text-sm text-ink-muted">{role.explanation}</p>
      {role.matching_skills.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {role.matching_skills.map((skill) => (
            <span key={skill} className="rounded-full bg-white px-2 py-0.5 text-xs text-ink-muted">
              {skill}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function RecommendedJobs({ resumeText }) {
  const [roles, setRoles] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleShowJobs() {
    setIsLoading(true);
    setError("");
    try {
      const data = await recommendJobs(resumeText);
      setRoles(data.recommended_roles);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not load recommended jobs.");
    } finally {
      setIsLoading(false);
    }
  }

  if (roles === null) {
    return (
      <div className="text-center">
        <button
          type="button"
          onClick={handleShowJobs}
          disabled={isLoading}
          className="inline-flex items-center gap-2 rounded-full border border-teal px-5 py-2.5 text-sm font-medium text-teal transition-colors hover:bg-teal-soft disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isLoading ? <Loader2 className="animate-spin" size={16} /> : <Briefcase size={16} />}
          {isLoading ? "Finding roles..." : "Show Recommended Jobs"}
        </button>
        {error && <p className="mt-2 text-sm text-critical">{error}</p>}
      </div>
    );
  }

  return (
    <div className="animate-unfold space-y-3">
      <h3 className="font-display text-lg font-semibold text-ink">Recommended jobs</h3>
      {roles.map((role) => (
        <RoleCard key={role.role_name} role={role} />
      ))}
    </div>
  );
}
