import { Loader2 } from "lucide-react";

export default function AnalyzeButton({ disabled, isLoading, onClick, helperText }) {
  return (
    <div className="flex flex-col items-center gap-2 pt-2">
      <button
        type="button"
        onClick={onClick}
        disabled={disabled || isLoading}
        className="flex w-full max-w-xs items-center justify-center gap-2 rounded-full bg-ink px-8 py-3 text-base font-medium text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-40 enabled:hover:opacity-90"
      >
        {isLoading && <Loader2 className="animate-spin" size={18} aria-hidden="true" />}
        {isLoading ? "Analyzing..." : "Analyze Resume"}
      </button>
      {helperText && !isLoading && (
        <p className="text-xs text-ink-muted">{helperText}</p>
      )}
    </div>
  );
}
