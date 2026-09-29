import { useRef, useState } from "react";
import { UploadCloud, FileText, X } from "lucide-react";

const ACCEPTED_EXTENSIONS = [".pdf", ".docx"];
const MAX_SIZE_MB = 10;

function isAcceptedFile(file) {
  const lowerName = file.name.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((ext) => lowerName.endsWith(ext));
}

/**
 * Drag-and-drop + browse file input. Validates extension and size
 * client-side for immediate feedback — the backend re-validates
 * regardless, so this is a UX convenience, not the source of truth.
 */
export default function FileDropzone({ file, onFileSelect, onRemove, compact = false }) {
  const inputRef = useRef(null);
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [localError, setLocalError] = useState("");

  function validateAndSelect(candidate) {
    if (!candidate) return;

    if (!isAcceptedFile(candidate)) {
      setLocalError("Only PDF and DOCX files are supported.");
      return;
    }
    if (candidate.size > MAX_SIZE_MB * 1024 * 1024) {
      setLocalError(`File exceeds the ${MAX_SIZE_MB} MB limit.`);
      return;
    }

    setLocalError("");
    onFileSelect(candidate);
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDraggingOver(false);
    validateAndSelect(event.dataTransfer.files?.[0]);
  }

  function handleBrowseChange(event) {
    validateAndSelect(event.target.files?.[0]);
    event.target.value = ""; // allow re-selecting the same file later
  }

  if (file) {
    return (
      <div className="flex items-center justify-between rounded border border-line bg-surface px-4 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <FileText className="shrink-0 text-teal" size={20} aria-hidden="true" />
          <div className="min-w-0">
            <p className="truncate text-sm font-medium text-ink">{file.name}</p>
            <p className="text-xs text-ink-muted">{(file.size / 1024).toFixed(0)} KB</p>
          </div>
        </div>
        <button
          type="button"
          onClick={onRemove}
          aria-label={`Remove ${file.name}`}
          className="ml-3 shrink-0 rounded p-1 text-ink-muted hover:bg-paper hover:text-ink"
        >
          <X size={18} />
        </button>
      </div>
    );
  }

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDraggingOver(true);
        }}
        onDragLeave={() => setIsDraggingOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded border-2 border-dashed text-center transition-colors ${
          compact ? "px-4 py-6" : "px-6 py-10"
        } ${
          isDraggingOver
            ? "border-teal bg-teal-soft"
            : "border-line bg-surface hover:border-teal"
        }`}
      >
        <UploadCloud className="text-ink-muted" size={compact ? 22 : 28} aria-hidden="true" />
        <p className="text-sm text-ink">
          <span className="font-medium text-teal">Browse</span> or drag a file here
        </p>
        <p className="text-xs text-ink-muted">PDF or DOCX, up to {MAX_SIZE_MB} MB</p>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx"
          className="hidden"
          onChange={handleBrowseChange}
        />
      </div>
      {localError && <p className="mt-2 text-sm text-critical">{localError}</p>}
    </div>
  );
}
