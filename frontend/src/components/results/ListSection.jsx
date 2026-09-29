import { Check, AlertCircle } from "lucide-react";

const VARIANTS = {
  positive: { icon: Check, color: "text-teal" },
  warning: { icon: AlertCircle, color: "text-amber" },
};

export default function ListSection({ title, items, variant = "positive", emptyText }) {
  const { icon: Icon, color } = VARIANTS[variant];

  return (
    <div>
      <h3 className="font-display text-lg font-semibold text-ink">{title}</h3>
      {items.length === 0 ? (
        <p className="mt-2 text-sm text-ink-muted">{emptyText}</p>
      ) : (
        <ul className="mt-2 space-y-2">
          {items.map((item, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-ink">
              <Icon className={`mt-0.5 shrink-0 ${color}`} size={16} aria-hidden="true" />
              <span>{item}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
