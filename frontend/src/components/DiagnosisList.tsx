import type { LocalizedDiagnosis } from '../types';
import DiagnosisCard from './DiagnosisCard';

interface DiagnosisListProps {
  diagnoses: LocalizedDiagnosis[];
}

export default function DiagnosisList({ diagnoses }: DiagnosisListProps) {
  // Sort by severity descending (already sorted by backend, but ensure client-side)
  const sorted = [...diagnoses].sort((a, b) => b.severity - a.severity);

  if (sorted.length === 0) {
    return (
      <div
        className="rounded-xl border px-4 py-6 text-center"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          borderColor: 'var(--border-light)',
        }}
      >
        <div
          className="mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-full"
          style={{ backgroundColor: 'var(--accent-50)' }}
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 20 20"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M6.5 10l2.5 2.5 5-5"
              stroke="var(--accent-600)"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          לא נמצאו אבחנות — הטקסט נראה תקין
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {sorted.map((diagnosis) => (
        <DiagnosisCard key={diagnosis.type} diagnosis={diagnosis} />
      ))}
    </div>
  );
}
