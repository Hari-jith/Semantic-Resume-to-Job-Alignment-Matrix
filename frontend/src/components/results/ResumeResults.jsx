import ScoreDisplay from "./ScoreDisplay";
import ScoreBreakdown from "./ScoreBreakdown";
import SkillsPanel from "./SkillsPanel";
import ListSection from "./ListSection";
import RecommendedJobs from "./RecommendedJobs";

export default function ResumeResults({ data }) {
  return (
    <div className="animate-unfold space-y-6 rounded border border-line bg-white/40 p-6">
      <div className="flex flex-col items-center gap-4 border-b border-line pb-6 sm:flex-row sm:items-start sm:justify-between">
        <div className="order-2 sm:order-1">
          {data.summary && (
            <div>
              <h3 className="font-display text-lg font-semibold text-ink">Resume summary</h3>
              <p className="mt-1 max-w-md text-sm text-ink-muted">{data.summary}</p>
            </div>
          )}
        </div>
        <div className="order-1 sm:order-2">
          <ScoreDisplay score={data.ats_score} label="ATS Score" />
        </div>
      </div>

      <ScoreBreakdown
        breakdown={data.score_breakdown}
        excludedComponents={data.excluded_score_components}
        notes={data.score_notes}
      />

      <div className="grid gap-6 border-t border-line pt-6 sm:grid-cols-2">
        <SkillsPanel skillsByCategory={data.skills_by_category} />
        <div className="space-y-6">
          <ListSection
            title="Strengths"
            items={data.strengths}
            variant="positive"
            emptyText="No specific strengths were identified from the resume text."
          />
          <ListSection
            title="Areas for improvement"
            items={data.areas_for_improvement}
            variant="warning"
            emptyText="No specific improvements were suggested."
          />
        </div>
      </div>

      <div className="border-t border-line pt-6">
        <RecommendedJobs resumeText={data.resume_text} />
      </div>
    </div>
  );
}
