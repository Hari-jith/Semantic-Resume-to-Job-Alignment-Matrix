export default function Header() {
  return (
    <header className="border-b border-line">
      <div className="mx-auto max-w-4xl px-6 pt-10 pb-8">
        <div className="mb-4 h-1.5 w-14 rounded-full bg-teal" aria-hidden="true" />
        <h1 className="font-display text-4xl font-semibold tracking-tight text-ink">
          Semantic Resume ATS
        </h1>
        <p className="mt-2 max-w-lg text-ink-muted">
          Upload your resume for a structured review. Add a job description
          and we'll check how well it lines up, skill by skill.
        </p>
      </div>
    </header>
  );
}
