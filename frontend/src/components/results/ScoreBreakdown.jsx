function formatLabel(key) {
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function barColor(score) {
  if (score >= 75) return "bg-teal";
  if (score >= 50) return "bg-amber";
  return "bg-critical";
}

export default function ScoreBreakdown({ breakdown, excludedComponents = [], notes }) {
  const entries = Object.entries(breakdown);

  return (
    <div>
      <h3 className="font-display text-lg font-semibold text-ink">Score breakdown</h3>
      <div className="mt-3 space-y-3">
        {entries.map(([key, { score, weight }]) => (
          <div key={key}>
            <div className="mb-1 flex items-baseline justify-between text-sm">
              <span className="text-ink">{formatLabel(key)}</span>
              <span className="text-ink-muted">
                {score.toFixed(0)} · {(weight * 100).toFixed(0)}% weight
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-line/50">
              <div
                className={`h-full rounded-full ${barColor(score)}`}
                style={{ width: `${Math.max(0, Math.min(100, score))}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {excludedComponents.length > 0 && (
        <p className="mt-3 text-xs text-ink-muted">
          Not included in this score (not enough information to assess):{" "}
          {excludedComponents.map(formatLabel).join(", ")}.
        </p>
      )}

      {notes && <p className="mt-4 text-sm text-ink-muted">{notes}</p>}
    </div>
  );
}
