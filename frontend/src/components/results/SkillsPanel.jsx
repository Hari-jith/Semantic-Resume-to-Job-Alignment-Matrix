export default function SkillsPanel({ skillsByCategory }) {
  const categories = Object.entries(skillsByCategory);

  if (categories.length === 0) {
    return (
      <div>
        <h3 className="font-display text-lg font-semibold text-ink">Detected skills</h3>
        <p className="mt-2 text-sm text-ink-muted">
          No skills from our taxonomy were detected. Try listing tools, languages, or frameworks explicitly.
        </p>
      </div>
    );
  }

  return (
    <div>
      <h3 className="font-display text-lg font-semibold text-ink">Detected skills</h3>
      <div className="mt-3 space-y-3">
        {categories.map(([category, skills]) => (
          <div key={category}>
            <p className="mb-1.5 text-xs font-medium uppercase tracking-wide text-ink-muted">
              {category}
            </p>
            <div className="flex flex-wrap gap-1.5">
              {skills.map((skill) => (
                <span
                  key={skill}
                  className="rounded-full bg-teal-soft px-2.5 py-1 text-xs font-medium text-teal"
                >
                  {skill}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
