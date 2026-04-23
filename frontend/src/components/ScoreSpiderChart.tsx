import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import type { ScoresResponse } from '../types';

const SCORE_AXES: { key: keyof ScoresResponse; label: string }[] = [
  { key: 'difficulty', label: 'קושי' },
  { key: 'style', label: 'סגנון' },
  { key: 'fluency', label: 'שטף' },
  { key: 'cohesion', label: 'קוהרנטיות' },
  { key: 'complexity', label: 'מורכבות' },
];

interface ScoreSpiderChartProps {
  scores: ScoresResponse;
  revisedScores?: ScoresResponse | null;
  isLoading?: boolean;
}

/* Custom tooltip that shows all 5 scores */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function AllScoresTooltip({ active, scores, revisedScores }: { active?: boolean; scores: ScoresResponse; revisedScores?: ScoresResponse | null }) {
  if (!active) return null;

  return (
    <div
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-light)',
        borderRadius: 10,
        boxShadow: 'var(--shadow-md)',
        fontFamily: "'Heebo', system-ui, sans-serif",
        fontSize: 13,
        direction: 'rtl',
        padding: '10px 14px',
        minWidth: 160,
      }}
    >
      {SCORE_AXES.map(({ key, label }) => {
        const val = scores[key];
        const revised = revisedScores?.[key];
        return (
          <div
            key={key}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: 12,
              padding: '3px 0',
            }}
          >
            <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>
              {label}
            </span>
            <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>
              {(val * 100).toFixed(0)}%
              {revised != null && (
                <span
                  style={{
                    marginRight: 6,
                    fontSize: 11,
                    color:
                      revised > val
                        ? 'var(--severity-low)'
                        : revised < val
                          ? 'var(--severity-high)'
                          : 'var(--text-muted)',
                  }}
                >
                  → {(revised * 100).toFixed(0)}%
                </span>
              )}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function ScoreSpiderChart({
  scores,
  revisedScores,
  isLoading,
}: ScoreSpiderChartProps) {
  const data = SCORE_AXES.map(({ key, label }) => ({
    axis: label,
    current: scores[key],
    revised: revisedScores ? revisedScores[key] : undefined,
  }));

  return (
    <div
      className="relative mx-auto"
      style={{
        width: '100%',
        maxWidth: 260,
        aspectRatio: '1',
        opacity: isLoading ? 0.35 : 1,
        filter: isLoading ? 'grayscale(0.8)' : 'none',
        transition: 'opacity 0.3s ease, filter 0.3s ease',
      }}
    >
      {isLoading && (
        <div className="absolute inset-0 z-10 flex items-center justify-center">
          <span
            className="inline-block h-7 w-7 animate-spin rounded-full border-3 border-current/20 border-t-current"
            style={{ color: 'var(--accent-600)' }}
          />
        </div>
      )}
      {/* Decorative ring behind the chart */}
      <div
        className="absolute inset-0 rounded-full opacity-[0.04]"
        style={{
          background:
            'radial-gradient(circle, var(--accent-500) 0%, transparent 70%)',
        }}
      />

      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid
            stroke="var(--border-light)"
            strokeDasharray="3 3"
          />
          <PolarAngleAxis
            dataKey="axis"
            tick={{
              fill: 'var(--text-secondary)',
              fontSize: 12,
              fontFamily: 'var(--font-body)',
              fontWeight: 500,
            }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 1]}
            tick={false}
            axisLine={false}
          />

          {/* Before scores (shown as overlay when comparing) */}
          {revisedScores && (
            <Radar
              name="מקורי"
              dataKey="current"
              stroke="var(--primary-400)"
              fill="var(--primary-300)"
              fillOpacity={0.15}
              strokeWidth={1.5}
              strokeDasharray="4 3"
            />
          )}

          {/* Current / revised scores */}
          <Radar
            name={revisedScores ? 'מתוקן' : 'ציונים'}
            dataKey={revisedScores ? 'revised' : 'current'}
            stroke="var(--accent-600)"
            fill="var(--accent-400)"
            fillOpacity={0.25}
            strokeWidth={2}
            dot={{
              r: 3,
              fill: 'var(--accent-600)',
              stroke: 'var(--bg-secondary)',
              strokeWidth: 1.5,
            }}
          />

          <Tooltip
            content={<AllScoresTooltip scores={scores} revisedScores={revisedScores} />}
            cursor={false}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
