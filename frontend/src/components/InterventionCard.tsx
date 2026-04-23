import { useState } from 'react';
import type { LocalizedIntervention } from '../types';
import { useAppStore } from '../store/useAppStore';
import { rewriteText } from '../api/client';
import PracticeModal from './PracticeModal';

interface InterventionCardProps {
  intervention: LocalizedIntervention;
}

function priorityColor(priority: number): string {
  if (priority > 0.7) return 'var(--severity-high)';
  if (priority > 0.4) return 'var(--severity-medium)';
  return 'var(--severity-low)';
}

export default function InterventionCard({
  intervention,
}: InterventionCardProps) {
  const { analysisResult, openRewriteModal } = useAppStore();
  const [isRewriting, setIsRewriting] = useState(false);
  const [isPracticeOpen, setIsPracticeOpen] = useState(false);
  const color = priorityColor(intervention.priority);

  const targetSentences = analysisResult?.sentences ?? [];
  const sentenceText = targetSentences.map((s) => s.text).join(' ');

  const handlePractice = () => {
    setIsPracticeOpen(true);
  };

  const handleAIRewrite = async () => {
    if (!analysisResult) return;
    setIsRewriting(true);
    try {
      const result = await rewriteText(
        sentenceText,
        intervention.target_diagnosis,
        '',
      );
      openRewriteModal(
        targetSentences,
        intervention.target_diagnosis,
        result.suggestion,
      );
    } catch {
      openRewriteModal(targetSentences, intervention.target_diagnosis);
    } finally {
      setIsRewriting(false);
    }
  };

  return (
    <>
      <div
        className="card-hover rounded-xl border p-4"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          borderColor: 'var(--border-light)',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        {/* Priority bar */}
        <div
          className="mb-3 h-1 w-full overflow-hidden rounded-full"
          style={{ backgroundColor: 'var(--warm-200)' }}
        >
          <div
            className="severity-bar-fill h-full rounded-full"
            style={{
              width: `${intervention.priority * 100}%`,
              backgroundColor: color,
            }}
          />
        </div>

        {/* Target diagnosis label */}
        <div className="mb-2 flex items-center gap-2">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ backgroundColor: color }}
          />
          <span
            className="text-xs font-medium"
            style={{ color: 'var(--text-muted)' }}
          >
            {intervention.target_diagnosis}
          </span>
        </div>

        {/* Actions */}
        {intervention.actions_he.length > 0 && (
          <div className="mb-3">
            <p
              className="mb-1.5 text-xs font-medium"
              style={{ color: 'var(--text-muted)' }}
            >
              פעולות מומלצות
            </p>
            <ul className="space-y-1">
              {intervention.actions_he.map((action, i) => (
                <li
                  key={i}
                  className="flex items-start gap-2 text-sm"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  <span
                    className="mt-1.5 block h-1.5 w-1.5 shrink-0 rounded-full"
                    style={{ backgroundColor: 'var(--accent-500)' }}
                  />
                  {action}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Exercises */}
        {intervention.exercises_he.length > 0 && (
          <div className="mb-4">
            <p
              className="mb-1.5 text-xs font-medium"
              style={{ color: 'var(--text-muted)' }}
            >
              תרגילים
            </p>
            <ul className="space-y-1">
              {intervention.exercises_he.map((exercise, i) => (
                <li
                  key={i}
                  className="flex items-start gap-2 text-sm"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  <span
                    className="mt-0.5 text-xs"
                    style={{ color: 'var(--accent-600)' }}
                  >
                    ✦
                  </span>
                  {exercise}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Action buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={handlePractice}
            className="flex-1 cursor-pointer rounded-lg border px-3 py-2 text-sm font-medium transition-all"
            style={{
              borderColor: 'var(--border-medium)',
              color: 'var(--text-secondary)',
              backgroundColor: 'transparent',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'var(--accent-500)';
              e.currentTarget.style.color = 'var(--accent-700)';
              e.currentTarget.style.backgroundColor = 'var(--accent-50)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'var(--border-medium)';
              e.currentTarget.style.color = 'var(--text-secondary)';
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            🎯 תרגול
          </button>

          <button
            onClick={handleAIRewrite}
            disabled={isRewriting}
            className="flex-1 cursor-pointer rounded-lg px-3 py-2 text-sm font-medium text-white transition-all disabled:opacity-60 disabled:cursor-not-allowed"
            style={{
              background:
                'linear-gradient(135deg, var(--accent-600), var(--accent-700))',
              boxShadow: 'var(--shadow-sm)',
            }}
            onMouseEnter={(e) => {
              if (!isRewriting) {
                e.currentTarget.style.background =
                  'linear-gradient(135deg, var(--accent-500), var(--accent-600))';
                e.currentTarget.style.boxShadow = 'var(--shadow-md)';
              }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background =
                'linear-gradient(135deg, var(--accent-600), var(--accent-700))';
              e.currentTarget.style.boxShadow = 'var(--shadow-sm)';
            }}
          >
            {isRewriting ? (
              <span className="flex items-center justify-center gap-1.5">
                <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                טוען...
              </span>
            ) : (
              '🤖 הצעת שכתוב מ-AI'
            )}
          </button>
        </div>
      </div>

      <PracticeModal
        isOpen={isPracticeOpen}
        diagnosisType={intervention.target_diagnosis}
        sentenceText={sentenceText}
        onClose={() => setIsPracticeOpen(false)}
      />
    </>
  );
}
