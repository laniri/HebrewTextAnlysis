import { useAppStore } from '../store/useAppStore';

/* ── Score Hebrew labels ── */
const SCORE_LABELS: Record<string, string> = {
  difficulty: 'קושי',
  style: 'סגנון',
  fluency: 'שטף',
  cohesion: 'קוהרנטיות',
  complexity: 'מורכבות',
};

const SCORE_KEYS = Object.keys(SCORE_LABELS);

function deltaColor(value: number): string {
  if (value > 0.005) return 'var(--severity-low)';
  if (value < -0.005) return 'var(--severity-high)';
  return 'var(--text-muted)';
}

function deltaArrow(value: number): string {
  if (value > 0.005) return '↑';
  if (value < -0.005) return '↓';
  return '–';
}

export default function ProgressFeedback() {
  const { analysisResult, previousAnalysis } = useAppStore();

  // Only show when we have both current and previous analysis
  if (!analysisResult || !previousAnalysis) return null;

  const currentScores = analysisResult.scores;
  const previousScores = previousAnalysis.scores;

  // Compute deltas
  const deltas: { key: string; label: string; before: number; after: number; delta: number }[] = [];
  for (const key of SCORE_KEYS) {
    const before = previousScores[key as keyof typeof previousScores];
    const after = currentScores[key as keyof typeof currentScores];
    const delta = after - before;
    if (Math.abs(delta) > 0.005) {
      deltas.push({ key, label: SCORE_LABELS[key], before, after, delta });
    }
  }

  // Find resolved diagnoses (were in previous, not in current)
  const previousTypes = new Set(previousAnalysis.diagnoses.map((d) => d.type));
  const currentTypes = new Set(analysisResult.diagnoses.map((d) => d.type));
  const resolvedDiagnoses = previousAnalysis.diagnoses.filter(
    (d) => previousTypes.has(d.type) && !currentTypes.has(d.type),
  );

  // Nothing to show
  if (deltas.length === 0 && resolvedDiagnoses.length === 0) return null;

  return (
    <div
      className="rounded-xl border overflow-hidden"
      style={{
        borderColor: 'var(--accent-200)',
        backgroundColor: 'var(--accent-50)',
      }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-2 px-4 py-3"
        style={{
          background: 'linear-gradient(135deg, var(--accent-50), color-mix(in srgb, var(--accent-100) 60%, white))',
        }}
      >
        <div
          className="flex h-7 w-7 items-center justify-center rounded-lg"
          style={{ backgroundColor: 'var(--accent-200)' }}
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path
              d="M2 8l4 4 8-8"
              stroke="var(--accent-700)"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
        <h4
          className="text-sm font-semibold"
          style={{
            fontFamily: 'var(--font-heading)',
            color: 'var(--accent-800)',
          }}
        >
          התקדמות
        </h4>
      </div>

      <div className="px-4 pb-4 pt-2 space-y-3">
        {/* Score deltas */}
        {deltas.length > 0 && (
          <div className="space-y-1.5">
            {deltas.map(({ key, label, before, after, delta }) => (
              <div
                key={key}
                className="flex items-center justify-between rounded-lg px-3 py-2 text-sm"
                style={{ backgroundColor: 'var(--bg-secondary)' }}
              >
                <span
                  className="font-medium"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  {label}
                </span>
                <div className="flex items-center gap-3">
                  <span
                    className="text-xs"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    {(before * 100).toFixed(0)}%
                  </span>
                  <span style={{ color: 'var(--text-muted)' }}>→</span>
                  <span
                    className="text-xs font-medium"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    {(after * 100).toFixed(0)}%
                  </span>
                  <span
                    className="flex items-center gap-0.5 rounded-md px-1.5 py-0.5 text-xs font-bold"
                    style={{
                      color: deltaColor(delta),
                      backgroundColor: `color-mix(in srgb, ${deltaColor(delta)} 10%, transparent)`,
                    }}
                  >
                    {deltaArrow(delta)}
                    {Math.abs(delta * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Resolved diagnoses */}
        {resolvedDiagnoses.length > 0 && (
          <div className="space-y-1.5">
            <p
              className="text-xs font-medium"
              style={{ color: 'var(--accent-700)' }}
            >
              אבחנות שנפתרו
            </p>
            {resolvedDiagnoses.map((d) => (
              <div
                key={d.type}
                className="flex items-center gap-2 rounded-lg px-3 py-2"
                style={{ backgroundColor: 'var(--bg-secondary)' }}
              >
                <span
                  className="flex h-5 w-5 items-center justify-center rounded-full"
                  style={{ backgroundColor: 'var(--accent-200)' }}
                >
                  <svg
                    width="11"
                    height="11"
                    viewBox="0 0 16 16"
                    fill="none"
                    aria-hidden="true"
                  >
                    <path
                      d="M2 8l4 4 8-8"
                      stroke="var(--accent-700)"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </span>
                <span
                  className="text-sm"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  {d.label_he}
                </span>
                <span
                  className="text-xs"
                  style={{ color: 'var(--accent-600)' }}
                >
                  — נפתרה
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
