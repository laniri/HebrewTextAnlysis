import { useEffect, useState } from 'react';
import type { ExampleSummary } from '../types';
import { listExamples, getExample } from '../api/client';
import { useAppStore } from '../store/useAppStore';

/* ── Category icons (emoji-based for simplicity) ── */
const CATEGORY_ICONS: Record<string, string> = {
  'ציוץ': '🐦',
  'כתבה חדשותית': '📰',
  'טקסט משפטי': '⚖️',
  'חיבור': '📝',
  'פוסט בבלוג': '💬',
};

function getCategoryIcon(category: string): string {
  return CATEGORY_ICONS[category] ?? '📄';
}

export default function ExampleSelector() {
  const { setText, analyzeText } = useAppStore();
  const [examples, setExamples] = useState<ExampleSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [hasError, setHasError] = useState(false);

  /* ── Fetch example list on mount ── */
  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    listExamples()
      .then((data) => {
        if (!cancelled) {
          setExamples(data);
          setHasError(false);
        }
      })
      .catch(() => {
        if (!cancelled) setHasError(true);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  /* ── Load example into editor ── */
  const handleSelect = async (id: string) => {
    setLoadingId(id);
    try {
      const example = await getExample(id);
      setText(example.text);
      // Trigger analysis immediately after loading
      await analyzeText(example.text);
    } catch {
      // Silently fail — user can retry
    } finally {
      setLoadingId(null);
    }
  };

  // Don't render if loading or error or no examples
  if (isLoading) {
    return (
      <div className="flex items-center gap-2 px-4 py-2 md:px-6">
        <span
          className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-current/30 border-t-current"
          style={{ color: 'var(--text-muted)' }}
        />
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
          טוען דוגמאות...
        </span>
      </div>
    );
  }

  if (hasError || examples.length === 0) return null;

  return (
    <div
      className="flex items-center gap-3 overflow-x-auto px-4 py-2.5 md:px-6"
      style={{
        borderBottom: '1px solid var(--border-light)',
        backgroundColor: 'color-mix(in srgb, var(--bg-primary) 80%, var(--warm-100))',
      }}
    >
      <span
        className="shrink-0 text-sm font-medium"
        style={{ color: 'var(--text-muted)' }}
      >
        דוגמאות:
      </span>
      <div className="flex items-center gap-2">
        {examples.map((ex) => {
          const isActive = loadingId === ex.id;
          return (
            <button
              key={ex.id}
              onClick={() => handleSelect(ex.id)}
              disabled={isActive}
              className="flex shrink-0 cursor-pointer items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-all disabled:cursor-wait disabled:opacity-60"
              style={{
                borderColor: 'var(--border-light)',
                backgroundColor: 'var(--bg-secondary)',
                color: 'var(--text-secondary)',
              }}
              onMouseEnter={(e) => {
                if (!isActive) {
                  e.currentTarget.style.borderColor = 'var(--accent-400)';
                  e.currentTarget.style.backgroundColor = 'var(--accent-50)';
                  e.currentTarget.style.color = 'var(--accent-700)';
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'var(--border-light)';
                e.currentTarget.style.backgroundColor = 'var(--bg-secondary)';
                e.currentTarget.style.color = 'var(--text-secondary)';
              }}
              title={ex.preview}
            >
              {isActive ? (
                <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-current/30 border-t-current" />
              ) : (
                <span>{getCategoryIcon(ex.category)}</span>
              )}
              {ex.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
