import ScoreDisplay from "./ScoreDisplay";
import ScoreBreakdown from "./ScoreBreakdown";
import ListSection from "./ListSection";
import ImprovementsPanel from "./ImprovementsPanel";

export default function MatchResults({ data }) {
  return (
    <div className="animate-unfold space-y-6 rounded border border-line bg-white/40 p-6">
      <div className="flex flex-col items-center gap-4 border-b border-line pb-6 sm:flex-row sm:items-start sm:justify-between">
        <div className="order-2 sm:order-1">
          {data.jd_job_title && (
            <div>
              <p className="text-xs uppercase tracking-wide text-ink-muted">Matched against</p>
              <h3 className="font-display text-lg font-semibold text-ink">{data.jd_job_title}</h3>
            </div>
          )}
        </div>
        <div className="order-1 sm:order-2">
          <ScoreDisplay score={data.ats_match_score} label="ATS Match Score" />
        </div>
      </div>

      <ScoreBreakdown
        breakdown={data.score_breakdown}
        excludedComponents={data.excluded_score_components}
        notes={data.score_notes}
      />

      <div className="grid gap-6 border-t border-line pt-6 sm:grid-cols-2">
        <ListSection
          title="Matching skills"
          items={data.matching_skills}
          variant="positive"
          emptyText="No overlapping skills were detected between your resume and this job description."
        />
        <ListSection
          title="Missing skills"
          items={data.missing_skills}
          variant="warning"
          emptyText="No important skills from the job description appear to be missing."
        />
      </div>

      <div className="border-t border-line pt-6">
        <ListSection
          title="Strengths"
          items={data.strengths}
          variant="positive"
          emptyText="No specific strengths stood out for this particular job description."
        />
      </div>

      <div className="border-t border-line pt-6">
        <ImprovementsPanel resumeText={data.resume_text} jdText={data.jd_text} />
      </div>
    </div>
  );
}
