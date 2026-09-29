const RADIUS = 54;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function tierColor(score) {
  if (score >= 75) return "var(--color-teal)";
  if (score >= 50) return "var(--color-amber)";
  return "var(--color-critical)";
}

export default function ScoreDisplay({ score, label }) {
  const clamped = Math.max(0, Math.min(100, score));
  const offset = CIRCUMFERENCE * (1 - clamped / 100);
  const color = tierColor(clamped);

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative h-36 w-36">
        <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
          <circle cx="60" cy="60" r={RADIUS} fill="none" stroke="var(--color-line)" strokeWidth="10" />
          <circle
            cx="60"
            cy="60"
            r={RADIUS}
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 0.6s ease-out" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-display text-3xl font-semibold text-ink">{clamped.toFixed(0)}</span>
          <span className="text-xs text-ink-muted">/ 100</span>
        </div>
      </div>
      {label && <p className="text-sm font-medium text-ink">{label}</p>}
    </div>
  );
}
