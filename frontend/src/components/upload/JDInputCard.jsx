import FileDropzone from "./FileDropzone";

const MIN_JD_TEXT_LENGTH = 20;

export default function JDInputCard({
  inputMethod,
  onInputMethodChange,
  jdText,
  onTextChange,
  jdFile,
  onFileSelect,
  onRemoveFile,
}) {
  const showTooShortWarning =
    inputMethod === "text" && jdText.trim().length > 0 && jdText.trim().length < MIN_JD_TEXT_LENGTH;

  return (
    <section className="rounded border border-line bg-white/40 p-5">
      <div className="mb-3 flex items-baseline gap-2 border-l-2 border-amber pl-3">
        <h2 className="font-display text-lg font-semibold text-ink">Job description</h2>
        <span className="rounded-full bg-amber-soft px-2 py-0.5 text-xs font-medium text-amber">
          Optional
        </span>
      </div>

      <p className="mb-3 text-sm text-ink-muted">
        Add one to get a match score against this specific role. Skip it for a
        general resume review instead.
      </p>

      <div className="mb-3 inline-flex rounded-full border border-line bg-surface p-0.5 text-sm">
        <button
          type="button"
          onClick={() => onInputMethodChange("text")}
          className={`rounded-full px-3 py-1 transition-colors ${
            inputMethod === "text" ? "bg-teal text-white" : "text-ink-muted hover:text-ink"
          }`}
        >
          Paste text
        </button>
        <button
          type="button"
          onClick={() => onInputMethodChange("file")}
          className={`rounded-full px-3 py-1 transition-colors ${
            inputMethod === "file" ? "bg-teal text-white" : "text-ink-muted hover:text-ink"
          }`}
        >
          Upload file
        </button>
      </div>

      {inputMethod === "text" ? (
        <div>
          <textarea
            value={jdText}
            onChange={(e) => onTextChange(e.target.value)}
            placeholder="Paste the job description here..."
            rows={6}
            className="w-full resize-y rounded border border-line bg-surface p-3 text-sm text-ink placeholder:text-ink-muted focus:border-teal"
          />
          {showTooShortWarning && (
            <p className="mt-1 text-xs text-amber">
              Add a little more detail ({MIN_JD_TEXT_LENGTH} characters minimum) so we can analyze it.
            </p>
          )}
        </div>
      ) : (
        <FileDropzone file={jdFile} onFileSelect={onFileSelect} onRemove={onRemoveFile} compact />
      )}
    </section>
  );
}
