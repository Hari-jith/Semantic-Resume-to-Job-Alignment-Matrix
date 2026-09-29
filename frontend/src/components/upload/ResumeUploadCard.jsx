import FileDropzone from "./FileDropzone";

export default function ResumeUploadCard({ file, onFileSelect, onRemove }) {
  return (
    <section className="rounded border border-line bg-white/40 p-5">
      <div className="mb-3 flex items-baseline gap-2 border-l-2 border-teal pl-3">
        <h2 className="font-display text-lg font-semibold text-ink">Resume</h2>
        <span className="text-xs text-ink-muted">Required</span>
      </div>
      <FileDropzone file={file} onFileSelect={onFileSelect} onRemove={onRemove} />
    </section>
  );
}
