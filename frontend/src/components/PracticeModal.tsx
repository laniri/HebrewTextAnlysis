import { useState, useEffect, useRef, useCallback } from 'react';
import { diff_match_patch } from 'diff-match-patch';
import type { ExerciseResponse, ExerciseOption } from '../types';
import { generateExercise } from '../api/client';
import { useAppStore } from '../store/useAppStore';

const dmp = new diff_match_patch();

interface PracticeModalProps {
  isOpen: boolean;
  diagnosisType: string;
  sentenceText: string;
  onClose: () => void;
}

/** Truncate text to ~maxLen chars at a sentence boundary. */
function truncateAtSentence(text: string, maxLen: number = 512): string {
  if (text.length <= maxLen) return text;
  const cut = text.slice(0, maxLen);
  const lastPeriod = Math.max(
    cut.lastIndexOf('.'),
    cut.lastIndexOf('!'),
    cut.lastIndexOf('?'),
  );
  if (lastPeriod > maxLen * 0.4) {
    return cut.slice(0, lastPeriod + 1) + ' …';
  }
  return cut + '…';
}

/** Render inline diff between original and option text. */
function DiffText({ original, revised }: { original: string; revised: string }) {
  const diffs = dmp.diff_main(original, revised);
  dmp.diff_cleanupSemantic(diffs);

  return (
    <span style={{ direction: 'rtl' }}>
      {diffs.map(([op, text], i) => {
        if (op === 0) {
          return (
            <span key={i} style={{ color: 'var(--text-primary)' }}>
              {text}
            </span>
          );
        }
        if (op === 1) {
          return (
            <span
              key={i}
              style={{
                backgroundColor: 'rgba(39, 171, 131, 0.2)',
                color: 'var(--accent-800)',
                borderRadius: '2px',
                padding: '0 1px',
              }}
            >
              {text}
            </span>
          );
        }
        // op === -1
        return (
          <span
            key={i}
            style={{
              backgroundColor: 'rgba(225, 45, 57, 0.15)',
              color: 'var(--severity-high)',
              borderRadius: '2px',
              padding: '0 1px',
              textDecoration: 'line-through',
            }}
          >
            {text}
          </span>
        );
      })}
    </span>
  );
}

function OptionCard({
  option,
  index,
  selected,
  revealed,
  originalText,
  onSelect,
}: {
  option: ExerciseOption;
  index: number;
  selected: boolean;
  revealed: boolean;
  originalText: string;
  onSelect: () => void;
}) {
  const letters = ['א', 'ב', 'ג'];

  let borderColor = 'var(--border-light)';
  let bgColor = 'var(--bg-secondary)';

  if (revealed) {
    if (option.is_correct) {
      borderColor = 'var(--severity-low)';
      bgColor = 'rgba(39, 171, 131, 0.08)';
    } else if (selected) {
      borderColor = 'var(--severity-high)';
      bgColor = 'rgba(225, 45, 57, 0.06)';
    }
  } else if (selected) {
    borderColor = 'var(--accent-500)';
    bgColor = 'var(--accent-50)';
  }

  const truncatedOriginal = truncateAtSentence(originalText);
  const truncatedOption = truncateAtSentence(option.text);

  return (
    <button
      onClick={onSelect}
      disabled={revealed}
      className="w-full cursor-pointer rounded-xl border-2 p-4 text-right transition-all disabled:cursor-default"
      style={{ borderColor, backgroundColor: bgColor }}
      onMouseEnter={(e) => {
        if (!revealed && !selected) {
          e.currentTarget.style.borderColor = 'var(--accent-400)';
          e.currentTarget.style.backgroundColor = 'var(--warm-50)';
        }
      }}
      onMouseLeave={(e) => {
        if (!revealed && !selected) {
          e.currentTarget.style.borderColor = 'var(--border-light)';
          e.currentTarget.style.backgroundColor = 'var(--bg-secondary)';
        }
      }}
    >
      <div className="flex items-start gap-3">
        <span
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-sm font-bold"
          style={{
            backgroundColor: selected
              ? 'var(--accent-500)'
              : 'var(--warm-100)',
            color: selected ? 'white' : 'var(--text-secondary)',
          }}
        >
          {letters[index]}
        </span>
        <div className="flex-1">
          <div className="text-sm leading-relaxed">
            <DiffText original={truncatedOriginal} revised={truncatedOption} />
          </div>
          {revealed && (
            <div
              className="mt-2 flex items-start gap-2 rounded-lg px-3 py-2 text-xs leading-relaxed"
              style={{
                backgroundColor: option.is_correct
                  ? 'rgba(39, 171, 131, 0.1)'
                  : 'rgba(225, 45, 57, 0.06)',
                color: option.is_correct
                  ? 'var(--accent-800)'
                  : 'var(--severity-high)',
              }}
            >
              <span className="mt-0.5">
                {option.is_correct ? '✓' : '✗'}
              </span>
              <span>{option.explanation_he}</span>
            </div>
          )}
        </div>
      </div>
    </button>
  );
}

export default function PracticeModal({
  isOpen,
  diagnosisType,
  sentenceText,
  onClose,
}: PracticeModalProps) {
  const [exercise, setExercise] = useState<ExerciseResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [revealed, setRevealed] = useState(false);
  const backdropRef = useRef<HTMLDivElement>(null);
  const { text, setText, analyzeText } = useAppStore();

  const loadExercise = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setSelectedIndex(null);
    setRevealed(false);
    try {
      const result = await generateExercise(sentenceText, diagnosisType);
      setExercise(result);
    } catch {
      setError('לא הצלחנו ליצור תרגיל כרגע. נסו שוב.');
    } finally {
      setIsLoading(false);
    }
  }, [sentenceText, diagnosisType]);

  useEffect(() => {
    if (isOpen && sentenceText && diagnosisType) {
      loadExercise();
    }
  }, [isOpen, sentenceText, diagnosisType, loadExercise]);

  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === backdropRef.current) onClose();
  };

  const handleCheck = () => {
    if (selectedIndex !== null) setRevealed(true);
  };

  const handleApplyCorrect = () => {
    if (!exercise) return;
    const correct = exercise.options.find((o) => o.is_correct);
    if (!correct) return;
    const idx = text.indexOf(sentenceText);
    if (idx !== -1) {
      const newText =
        text.slice(0, idx) + correct.text + text.slice(idx + sentenceText.length);
      setText(newText);
      analyzeText(newText);
    }
    onClose();
  };

  if (!isOpen) return null;

  const isCorrectSelected =
    revealed &&
    selectedIndex !== null &&
    exercise?.options[selectedIndex]?.is_correct;

  return (
    <div
      ref={backdropRef}
      onClick={handleBackdropClick}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 max-md:p-0"
      style={{
        backgroundColor: 'rgba(16, 42, 67, 0.5)',
        backdropFilter: 'blur(4px)',
      }}
      role="dialog"
      aria-modal="true"
      aria-label="תרגול"
    >
      <div
        className="flex w-full max-w-2xl flex-col overflow-hidden rounded-2xl max-md:h-full max-md:max-w-none max-md:rounded-none"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          boxShadow: 'var(--shadow-xl)',
          maxHeight: 'min(90vh, 750px)',
        }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between border-b px-5 py-4"
          style={{ borderColor: 'var(--border-light)' }}
        >
          <div className="flex items-center gap-3">
            <div
              className="flex h-9 w-9 items-center justify-center rounded-xl"
              style={{
                background: 'linear-gradient(135deg, var(--primary-100), var(--primary-200))',
              }}
            >
              <span className="text-lg">🎯</span>
            </div>
            <div>
              <h2
                className="text-base font-semibold"
                style={{ fontFamily: 'var(--font-heading)', color: 'var(--text-primary)' }}
              >
                תרגול — בחרו את השכתוב הטוב ביותר
              </h2>
              {exercise && (
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {exercise.diagnosis_label_he}
                </p>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-lg border-none transition-colors"
            style={{ backgroundColor: 'transparent', color: 'var(--text-muted)' }}
            aria-label="סגור"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M12 4L4 12M4 4l8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">
          {isLoading && (
            <div className="flex flex-col items-center justify-center py-12">
              <span
                className="mb-3 inline-block h-8 w-8 animate-spin rounded-full border-3 border-current/20 border-t-current"
                style={{ color: 'var(--accent-600)' }}
              />
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>יוצר תרגיל...</p>
            </div>
          )}

          {error && (
            <div className="flex flex-col items-center py-8 text-center">
              <p className="mb-4 text-sm" style={{ color: 'var(--severity-high)' }}>{error}</p>
              <button
                onClick={loadExercise}
                className="cursor-pointer rounded-lg px-4 py-2 text-sm font-medium text-white"
                style={{ background: 'linear-gradient(135deg, var(--accent-600), var(--accent-700))', border: 'none' }}
              >
                נסו שוב
              </button>
            </div>
          )}

          {exercise && !isLoading && !error && (
            <>
              {/* Original text */}
              <section>
                <label className="mb-2 block text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                  הטקסט המקורי
                </label>
                <div
                  className="rounded-xl border px-4 py-3 text-sm leading-relaxed"
                  style={{
                    backgroundColor: 'var(--warm-50)',
                    borderColor: 'var(--border-light)',
                    color: 'var(--text-secondary)',
                    direction: 'rtl',
                  }}
                >
                  {truncateAtSentence(exercise.original_text)}
                </div>
              </section>

              {/* Diagnosis info */}
              <div
                className="rounded-lg px-4 py-3"
                style={{ backgroundColor: 'var(--primary-50)', borderRight: '3px solid var(--primary-400)' }}
              >
                <p className="text-sm font-medium" style={{ color: 'var(--primary-800)' }}>
                  {exercise.diagnosis_label_he}
                </p>
                <p className="mt-1 text-xs" style={{ color: 'var(--text-muted)' }}>
                  {exercise.diagnosis_explanation_he}
                </p>
                <p className="mt-1 text-xs" style={{ color: 'var(--accent-700)' }}>
                  💡 {exercise.tip_he}
                </p>
              </div>

              {/* Diff legend */}
              <div className="flex items-center gap-4 text-xs" style={{ color: 'var(--text-muted)' }}>
                <span className="flex items-center gap-1">
                  <span style={{ backgroundColor: 'rgba(39, 171, 131, 0.2)', padding: '1px 4px', borderRadius: '2px', color: 'var(--accent-800)' }}>טקסט</span>
                  נוסף
                </span>
                <span className="flex items-center gap-1">
                  <span style={{ backgroundColor: 'rgba(225, 45, 57, 0.15)', padding: '1px 4px', borderRadius: '2px', color: 'var(--severity-high)', textDecoration: 'line-through' }}>טקסט</span>
                  הוסר
                </span>
              </div>

              {/* Options */}
              <section>
                <label className="mb-3 block text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                  איזה שכתוב מתקן את הבעיה?
                </label>
                <div className="space-y-3">
                  {exercise.options.map((option, i) => (
                    <OptionCard
                      key={i}
                      option={option}
                      index={i}
                      selected={selectedIndex === i}
                      revealed={revealed}
                      originalText={exercise.original_text}
                      onSelect={() => { if (!revealed) setSelectedIndex(i); }}
                    />
                  ))}
                </div>
              </section>

              {/* Result feedback */}
              {revealed && (
                <div
                  className="rounded-xl border px-4 py-3 text-center"
                  style={{
                    borderColor: isCorrectSelected ? 'var(--accent-300)' : 'var(--severity-medium)',
                    backgroundColor: isCorrectSelected ? 'var(--accent-50)' : 'rgba(240, 180, 41, 0.08)',
                  }}
                >
                  <p className="text-sm font-semibold" style={{ color: isCorrectSelected ? 'var(--accent-800)' : 'var(--text-primary)' }}>
                    {isCorrectSelected ? '🎉 כל הכבוד! בחרתם נכון!' : '🤔 לא בדיוק — עיינו בהסברים למטה'}
                  </p>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div
          className="flex items-center justify-between border-t px-5 py-4"
          style={{ borderColor: 'var(--border-light)' }}
        >
          <button
            onClick={onClose}
            className="cursor-pointer rounded-lg border px-5 py-2.5 text-sm font-medium"
            style={{ borderColor: 'var(--border-medium)', color: 'var(--text-secondary)', backgroundColor: 'transparent' }}
          >
            סגור
          </button>
          <div className="flex items-center gap-2">
            {revealed && (
              <button
                onClick={loadExercise}
                className="cursor-pointer rounded-lg border px-4 py-2.5 text-sm font-medium"
                style={{ borderColor: 'var(--accent-400)', color: 'var(--accent-700)', backgroundColor: 'var(--accent-50)' }}
              >
                🔄 תרגיל נוסף
              </button>
            )}
            {revealed && isCorrectSelected && (
              <button
                onClick={handleApplyCorrect}
                className="cursor-pointer rounded-lg px-5 py-2.5 text-sm font-medium text-white"
                style={{ background: 'linear-gradient(135deg, var(--accent-600), var(--accent-700))', boxShadow: 'var(--shadow-sm)', border: 'none' }}
              >
                החל על הטקסט
              </button>
            )}
            {!revealed && (
              <button
                onClick={handleCheck}
                disabled={selectedIndex === null}
                className="cursor-pointer rounded-lg px-5 py-2.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
                style={{ background: 'linear-gradient(135deg, var(--primary-700), var(--primary-800))', boxShadow: 'var(--shadow-sm)', border: 'none' }}
              >
                בדיקה
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
