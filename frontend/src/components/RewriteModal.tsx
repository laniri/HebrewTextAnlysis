import { useEffect, useRef, useState, useCallback } from 'react';
import { diff_match_patch } from 'diff-match-patch';
import { useAppStore } from '../store/useAppStore';
import { reviseText } from '../api/client';

const dmp = new diff_match_patch();

/* ── Score Hebrew labels ── */
const SCORE_LABELS: Record<string, string> = {
  difficulty: 'קושי',
  style: 'סגנון',
  fluency: 'שטף',
  cohesion: 'קוהרנטיות',
  complexity: 'מורכבות',
};

/* ── Delta badge color helper ── */
function deltaColor(value: number): string {
  if (value > 0) return 'var(--severity-low)';
  if (value < 0) return 'var(--severity-high)';
  return 'var(--text-muted)';
}

function deltaSign(value: number): string {
  if (value > 0) return '+';
  return '';
}

/* ── Diff rendering ── */
interface DiffSegment {
  type: 'equal' | 'insert' | 'delete';
  text: string;
}

function computeDiff(original: string, revised: string): DiffSegment[] {
  const diffs = dmp.diff_main(original, revised);
  dmp.diff_cleanupSemantic(diffs);
  return diffs.map(([op, text]) => ({
    type: op === 0 ? 'equal' : op === 1 ? 'insert' : 'delete',
    text,
  }));
}

export default function RewriteModal() {
  const {
    rewriteModal,
    closeRewriteModal,
    updateRevisedText,
    setDeltaScores,
    applyRewrite,
  } = useAppStore();

  const [isRevising, setIsRevising] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backdropRef = useRef<HTMLDivElement>(null);

  const { isOpen, targetSentences, diagnosisType, revisedText, deltaScores } =
    rewriteModal;

  const originalText = targetSentences.map((s) => s.text).join(' ');

  /* ── Auto-focus textarea on open ── */
  useEffect(() => {
    if (isOpen && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isOpen]);

  /* ── Debounced revise call ── */
  const handleTextChange = useCallback(
    (newText: string) => {
      updateRevisedText(newText);

      if (debounceRef.current) clearTimeout(debounceRef.current);

      debounceRef.current = setTimeout(async () => {
        if (!newText.trim() || newText === originalText) {
          setDeltaScores(null);
          return;
        }
        setIsRevising(true);
        try {
          const result = await reviseText(originalText, newText);
          setDeltaScores(result.deltas);
        } catch {
          setDeltaScores(null);
        } finally {
          setIsRevising(false);
        }
      }, 600);
    },
    [originalText, updateRevisedText, setDeltaScores],
  );

  /* ── Cleanup debounce on unmount ── */
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  /* ── Close on Escape ── */
  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeRewriteModal();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, closeRewriteModal]);

  /* ── Close on backdrop click ── */
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === backdropRef.current) closeRewriteModal();
  };

  if (!isOpen) return null;

  const diffSegments = computeDiff(originalText, revisedText);

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
      aria-label="חלון שכתוב"
    >
      <div
        className="flex w-full max-w-2xl flex-col overflow-hidden rounded-2xl max-md:h-full max-md:max-w-none max-md:rounded-none"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          boxShadow: 'var(--shadow-xl)',
          maxHeight: 'min(90vh, 720px)',
        }}
      >
        {/* ── Header ── */}
        <div
          className="flex items-center justify-between border-b px-5 py-4"
          style={{ borderColor: 'var(--border-light)' }}
        >
          <div className="flex items-center gap-3">
            <div
              className="flex h-9 w-9 items-center justify-center rounded-xl"
              style={{
                background: 'linear-gradient(135deg, var(--accent-100), var(--accent-200))',
              }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path
                  d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"
                  stroke="var(--accent-700)"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M18.5 2.5a2.121 2.121 0 113 3L12 15l-4 1 1-4 9.5-9.5z"
                  stroke="var(--accent-700)"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
            <div>
              <h2
                className="text-base font-semibold"
                style={{
                  fontFamily: 'var(--font-heading)',
                  color: 'var(--text-primary)',
                }}
              >
                שכתוב מונחה
              </h2>
              {diagnosisType && (
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {diagnosisType}
                </p>
              )}
            </div>
          </div>
          <button
            onClick={closeRewriteModal}
            className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-lg border-none transition-colors"
            style={{
              backgroundColor: 'transparent',
              color: 'var(--text-muted)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--warm-100)';
              e.currentTarget.style.color = 'var(--text-secondary)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.color = 'var(--text-muted)';
            }}
            aria-label="סגור"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path
                d="M12 4L4 12M4 4l8 8"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </div>

        {/* ── Scrollable body ── */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">
          {/* Original text */}
          <section>
            <label
              className="mb-2 block text-xs font-medium"
              style={{ color: 'var(--text-muted)' }}
            >
              טקסט מקורי
            </label>
            <div
              className="rounded-xl border px-4 py-3 text-sm leading-relaxed"
              style={{
                backgroundColor: 'var(--warm-50)',
                borderColor: 'var(--border-light)',
                color: 'var(--text-secondary)',
              }}
            >
              {originalText}
            </div>
          </section>

          {/* Editable rewrite area */}
          <section>
            <label
              htmlFor="rewrite-textarea"
              className="mb-2 block text-xs font-medium"
              style={{ color: 'var(--text-muted)' }}
            >
              הגרסה המתוקנת
            </label>
            <textarea
              id="rewrite-textarea"
              ref={textareaRef}
              value={revisedText}
              onChange={(e) => handleTextChange(e.target.value)}
              rows={4}
              className="w-full resize-y rounded-xl border px-4 py-3 text-sm leading-relaxed outline-none transition-colors"
              style={{
                backgroundColor: 'var(--bg-editor)',
                borderColor: 'var(--border-light)',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-body)',
                direction: 'rtl',
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = 'var(--accent-400)';
                e.currentTarget.style.boxShadow = '0 0 0 3px color-mix(in srgb, var(--accent-400) 15%, transparent)';
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = 'var(--border-light)';
                e.currentTarget.style.boxShadow = 'none';
              }}
              placeholder="כתבו כאן את הגרסה המשופרת..."
            />
          </section>

          {/* Diff view */}
          {revisedText !== originalText && (
            <section>
              <label
                className="mb-2 flex items-center gap-2 text-xs font-medium"
                style={{ color: 'var(--text-muted)' }}
              >
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path
                    d="M8 3v10M3 8h10"
                    stroke="currentColor"
                    strokeWidth="1.2"
                    strokeLinecap="round"
                  />
                </svg>
                השוואה
              </label>
              <div
                className="rounded-xl border px-4 py-3 text-sm leading-loose"
                style={{
                  backgroundColor: 'var(--bg-secondary)',
                  borderColor: 'var(--border-light)',
                  direction: 'rtl',
                }}
              >
                {diffSegments.map((seg, i) => {
                  if (seg.type === 'equal') {
                    return (
                      <span key={i} style={{ color: 'var(--text-primary)' }}>
                        {seg.text}
                      </span>
                    );
                  }
                  if (seg.type === 'insert') {
                    return (
                      <span
                        key={i}
                        style={{
                          backgroundColor: 'rgba(39, 171, 131, 0.15)',
                          color: 'var(--accent-800)',
                          borderRadius: '3px',
                          padding: '0 2px',
                          textDecoration: 'none',
                        }}
                      >
                        {seg.text}
                      </span>
                    );
                  }
                  /* delete */
                  return (
                    <span
                      key={i}
                      style={{
                        backgroundColor: 'rgba(225, 45, 57, 0.12)',
                        color: 'var(--severity-high)',
                        borderRadius: '3px',
                        padding: '0 2px',
                        textDecoration: 'line-through',
                      }}
                    >
                      {seg.text}
                    </span>
                  );
                })}
              </div>
            </section>
          )}

          {/* Delta scores */}
          {(deltaScores || isRevising) && (
            <section>
              <label
                className="mb-2 block text-xs font-medium"
                style={{ color: 'var(--text-muted)' }}
              >
                שינוי בציונים
              </label>
              {isRevising ? (
                <div
                  className="flex items-center gap-2 rounded-xl border px-4 py-3 text-sm"
                  style={{
                    backgroundColor: 'var(--accent-50)',
                    borderColor: 'var(--accent-200)',
                    color: 'var(--accent-700)',
                  }}
                >
                  <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-current/30 border-t-current" />
                  מחשב שינויים...
                </div>
              ) : (
                deltaScores && (
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(SCORE_LABELS).map(([key, label]) => {
                      const delta = deltaScores[key] ?? 0;
                      if (Math.abs(delta) < 0.001) return null;
                      const pct = (delta * 100).toFixed(1);
                      return (
                        <div
                          key={key}
                          className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium"
                          style={{
                            borderColor: 'var(--border-light)',
                            backgroundColor: 'var(--bg-secondary)',
                          }}
                        >
                          <span style={{ color: 'var(--text-secondary)' }}>
                            {label}
                          </span>
                          <span
                            className="font-bold"
                            style={{ color: deltaColor(delta) }}
                          >
                            {deltaSign(delta)}{pct}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )
              )}
            </section>
          )}
        </div>

        {/* ── Footer actions ── */}
        <div
          className="flex items-center justify-end gap-3 border-t px-5 py-4"
          style={{ borderColor: 'var(--border-light)' }}
        >
          <button
            onClick={closeRewriteModal}
            className="cursor-pointer rounded-lg border px-5 py-2.5 text-sm font-medium transition-all"
            style={{
              borderColor: 'var(--border-medium)',
              color: 'var(--text-secondary)',
              backgroundColor: 'transparent',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'var(--primary-400)';
              e.currentTarget.style.backgroundColor = 'var(--warm-50)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'var(--border-medium)';
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            בטל
          </button>
          <button
            onClick={() => {
              applyRewrite();
            }}
            disabled={!revisedText.trim() || revisedText === originalText}
            className="cursor-pointer rounded-lg px-5 py-2.5 text-sm font-medium text-white transition-all disabled:cursor-not-allowed disabled:opacity-50"
            style={{
              background: 'linear-gradient(135deg, var(--accent-600), var(--accent-700))',
              boxShadow: 'var(--shadow-sm)',
            }}
            onMouseEnter={(e) => {
              if (!e.currentTarget.disabled) {
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
            החל
          </button>
        </div>
      </div>
    </div>
  );
}
