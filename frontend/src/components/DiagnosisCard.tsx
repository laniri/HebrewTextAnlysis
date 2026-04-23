import type { LocalizedDiagnosis } from '../types';

interface DiagnosisCardProps {
  diagnosis: LocalizedDiagnosis;
}

function severityColor(severity: number): string {
  if (severity > 0.7) return 'var(--severity-high)';
  if (severity > 0.4) return 'var(--severity-medium)';
  return 'var(--severity-low)';
}

function severityLabel(severity: number): string {
  if (severity > 0.7) return 'גבוהה';
  if (severity > 0.4) return 'בינונית';
  return 'נמוכה';
}

export default function DiagnosisCard({ diagnosis }: DiagnosisCardProps) {
  const color = severityColor(diagnosis.severity);
  const label = severityLabel(diagnosis.severity);

  return (
    <div
      className="card-hover rounded-xl border p-4"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        borderColor: 'var(--border-light)',
        borderRightWidth: 3,
        borderRightColor: color,
        boxShadow: 'var(--shadow-sm)',
      }}
    >
      {/* Header: label + severity */}
      <div className="mb-3 flex items-center justify-between gap-3">
        <h4
          className="text-sm font-semibold"
          style={{
            fontFamily: 'var(--font-heading)',
            color: 'var(--text-primary)',
          }}
        >
          {diagnosis.label_he}
        </h4>
        <span
          className="shrink-0 rounded-full px-2.5 py-0.5 text-xs font-medium"
          style={{
            backgroundColor: `color-mix(in srgb, ${color} 12%, transparent)`,
            color,
          }}
        >
          {label} ({(diagnosis.severity * 100).toFixed(0)}%)
        </span>
      </div>

      {/* Severity bar */}
      <div
        className="mb-3 h-1.5 w-full overflow-hidden rounded-full"
        style={{ backgroundColor: 'var(--warm-200)' }}
      >
        <div
          className="severity-bar-fill h-full rounded-full"
          style={{
            width: `${diagnosis.severity * 100}%`,
            backgroundColor: color,
          }}
        />
      </div>

      {/* Explanation */}
      <p
        className="mb-3 text-sm leading-relaxed"
        style={{ color: 'var(--text-secondary)' }}
      >
        {diagnosis.explanation_he}
      </p>

      {/* Actions */}
      {diagnosis.actions_he.length > 0 && (
        <ul className="mb-3 space-y-1.5">
          {diagnosis.actions_he.map((action, i) => (
            <li
              key={i}
              className="flex items-start gap-2 text-sm"
              style={{ color: 'var(--text-secondary)' }}
            >
              <span
                className="mt-1.5 block h-1.5 w-1.5 shrink-0 rounded-full"
                style={{ backgroundColor: color }}
              />
              {action}
            </li>
          ))}
        </ul>
      )}

      {/* Tip */}
      {diagnosis.tip_he && (
        <div
          className="rounded-lg px-3 py-2 text-xs leading-relaxed"
          style={{
            backgroundColor: 'var(--warm-50)',
            color: 'var(--text-muted)',
            borderRight: `2px solid var(--warm-300)`,
          }}
        >
          <span className="font-medium" style={{ color: 'var(--text-secondary)' }}>
            💡 טיפ:{' '}
          </span>
          {diagnosis.tip_he}
        </div>
      )}
    </div>
  );
}
