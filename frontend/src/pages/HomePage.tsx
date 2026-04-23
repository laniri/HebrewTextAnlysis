import { useState } from 'react';
import MainLayout from '../components/MainLayout';
import MonacoEditor from '../components/MonacoEditor';
import ScoreSpiderChart from '../components/ScoreSpiderChart';
import DiagnosisList from '../components/DiagnosisList';
import InterventionCard from '../components/InterventionCard';
import ProgressFeedback from '../components/ProgressFeedback';
import ExampleSelector from '../components/ExampleSelector';
import { useAppStore } from '../store/useAppStore';
import { Link } from 'react-router-dom';

/* ── Empty-state placeholder for the editor pane ── */
function EditorEmptyOverlay() {
  const { text } = useAppStore();
  if (text.trim()) return null;

  return (
    <div className="flex flex-col items-center justify-center p-8 text-center"
      style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', pointerEvents: 'none', opacity: 0.7 }}>
      <div
        className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl"
        style={{ backgroundColor: 'var(--accent-50)' }}
      >
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M12 20h9M16.5 3.5a2.121 2.121 0 113 3L7 19l-4 1 1-4L16.5 3.5z"
            stroke="var(--accent-600)"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
      <h2
        className="mb-2 text-xl"
        style={{ fontFamily: 'var(--font-heading)', color: 'var(--text-primary)' }}
      >
        התחילו לכתוב
      </h2>
      <p className="max-w-sm text-sm" style={{ color: 'var(--text-muted)' }}>
        הקלידו או הדביקו טקסט בעברית כאן כדי לקבל ניתוח לשוני מיידי עם ציונים, אבחנות והמלצות לשיפור
      </p>
    </div>
  );
}

/* ── Editor pane with Monaco + empty overlay ── */
function EditorPane() {
  return (
    <div className="h-full">
      <MonacoEditor />
    </div>
  );
}

/* ── Loading skeleton for the analysis panel ── */
function AnalysisLoading() {
  return (
    <div className="space-y-4">
      {/* Spider chart skeleton */}
      <div className="flex justify-center">
        <div
          className="loading-pulse h-48 w-48 rounded-full"
          style={{ backgroundColor: 'var(--warm-100)' }}
        />
      </div>
      {/* Card skeletons */}
      {[1, 2].map((i) => (
        <div
          key={i}
          className="loading-pulse rounded-xl border p-4"
          style={{
            backgroundColor: 'var(--bg-secondary)',
            borderColor: 'var(--border-light)',
          }}
        >
          <div
            className="mb-3 h-4 w-1/3 rounded"
            style={{ backgroundColor: 'var(--warm-200)' }}
          />
          <div
            className="mb-2 h-1.5 w-full rounded-full"
            style={{ backgroundColor: 'var(--warm-100)' }}
          />
          <div
            className="h-3 w-2/3 rounded"
            style={{ backgroundColor: 'var(--warm-100)' }}
          />
        </div>
      ))}
    </div>
  );
}

/* ── Empty state for analysis panel ── */
function AnalysisEmpty() {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div
        className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl"
        style={{ backgroundColor: 'var(--primary-50)' }}
      >
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            stroke="var(--primary-400)"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
      <h3
        className="mb-2 text-lg"
        style={{ fontFamily: 'var(--font-heading)', color: 'var(--text-primary)' }}
      >
        ניתוח לשוני
      </h3>
      <p className="max-w-xs text-sm" style={{ color: 'var(--text-muted)' }}>
        הניתוח יופיע כאן לאחר הזנת טקסט — ציונים, אבחנות, והמלצות לשיפור הכתיבה
      </p>
    </div>
  );
}

/* ── Mobile collapsible section ── */
function MobileAccordionSection({
  title,
  badge,
  children,
  defaultOpen = false,
}: {
  title: string;
  badge?: number;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="md:contents">
      {/* Accordion header — visible only on mobile */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full cursor-pointer items-center justify-between rounded-lg border px-4 py-3 text-right md:hidden"
        style={{
          backgroundColor: isOpen ? 'var(--accent-50)' : 'var(--bg-secondary)',
          borderColor: isOpen ? 'var(--accent-200)' : 'var(--border-light)',
          border: '1px solid',
        }}
        aria-expanded={isOpen}
      >
        <div className="flex items-center gap-2">
          <h3
            className="text-sm font-semibold"
            style={{
              fontFamily: 'var(--font-heading)',
              color: 'var(--text-primary)',
            }}
          >
            {title}
          </h3>
          {badge !== undefined && badge > 0 && (
            <span
              className="inline-flex h-5 w-5 items-center justify-center rounded-full text-xs font-bold text-white"
              style={{ backgroundColor: 'var(--primary-500)' }}
            >
              {badge}
            </span>
          )}
        </div>
        <svg
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="none"
          className="shrink-0 transition-transform"
          style={{
            transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'var(--transition-base)',
          }}
          aria-hidden="true"
        >
          <path
            d="M4 6l4 4 4-4"
            stroke="var(--text-muted)"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>

      {/* Content — always visible on desktop, collapsible on mobile */}
      <div
        className="md:block"
        style={{
          display: undefined,
        }}
      >
        <div className="hidden md:block">{children}</div>
        <div
          className="md:hidden overflow-hidden transition-all"
          style={{
            maxHeight: isOpen ? '2000px' : '0',
            opacity: isOpen ? 1 : 0,
            transition: 'max-height 0.3s ease, opacity 0.2s ease',
          }}
        >
          <div className="pt-2">{children}</div>
        </div>
      </div>
    </div>
  );
}

/* ── Full analysis panel with real data ── */
function AnalysisPanel() {
  const { analysisResult, isAnalyzing } = useAppStore();

  if (isAnalyzing && !analysisResult) {
    return <AnalysisLoading />;
  }

  if (!analysisResult) {
    return <AnalysisEmpty />;
  }

  return (
    <div className="space-y-6">
      {/* Loading overlay when re-analyzing */}
      {isAnalyzing && (
        <div
          className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm"
          style={{
            backgroundColor: 'var(--accent-50)',
            color: 'var(--accent-700)',
          }}
        >
          <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-current/30 border-t-current" />
          מנתח מחדש...
        </div>
      )}

      {/* ── Spider Chart — always visible ── */}
      <MobileAccordionSection title="פרופיל לשוני" defaultOpen>
        <section>
          <h3
            className="mb-3 hidden text-base font-semibold md:block"
            style={{
              fontFamily: 'var(--font-heading)',
              color: 'var(--text-primary)',
            }}
          >
            פרופיל לשוני
          </h3>
          <div
            className="rounded-xl border p-4"
            style={{
              backgroundColor: 'var(--bg-secondary)',
              borderColor: 'var(--border-light)',
              boxShadow: 'var(--shadow-sm)',
            }}
          >
            <ScoreSpiderChart scores={analysisResult.scores} isLoading={isAnalyzing} />
          </div>
        </section>
      </MobileAccordionSection>

      {/* ── Methodology link ── */}
      <Link
        to="/methodology"
        className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium no-underline transition-all"
        style={{
          color: 'var(--accent-700)',
          backgroundColor: 'var(--accent-50)',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = 'var(--accent-100)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = 'var(--accent-50)';
        }}
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path
            d="M8 14A6 6 0 108 2a6 6 0 000 12zM8 5v3m0 2.5h.01"
            stroke="currentColor"
            strokeWidth="1.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        על המתודולוגיה
      </Link>

      {/* ── Diagnoses ── */}
      <MobileAccordionSection
        title="אבחנות"
        badge={analysisResult.diagnoses.length}
      >
        <section>
          <h3
            className="mb-3 hidden text-base font-semibold md:block"
            style={{
              fontFamily: 'var(--font-heading)',
              color: 'var(--text-primary)',
            }}
          >
            אבחנות
            {analysisResult.diagnoses.length > 0 && (
              <span
                className="mr-2 inline-flex h-5 w-5 items-center justify-center rounded-full text-xs font-bold text-white"
                style={{ backgroundColor: 'var(--primary-500)' }}
              >
                {analysisResult.diagnoses.length}
              </span>
            )}
          </h3>
          <DiagnosisList diagnoses={analysisResult.diagnoses} />
        </section>
      </MobileAccordionSection>

      {/* ── Interventions ── */}
      {analysisResult.interventions.length > 0 && (
        <MobileAccordionSection title="המלצות לשיפור">
          <section>
            <h3
              className="mb-3 hidden text-base font-semibold md:block"
              style={{
                fontFamily: 'var(--font-heading)',
                color: 'var(--text-primary)',
              }}
            >
              המלצות לשיפור
            </h3>
            <div className="space-y-3">
              {analysisResult.interventions.map((intervention) => (
                <InterventionCard
                  key={`${intervention.type}-${intervention.target_diagnosis}`}
                  intervention={intervention}
                />
              ))}
            </div>
          </section>
        </MobileAccordionSection>
      )}

      {/* ── Progress Feedback ── */}
      <ProgressFeedback />
    </div>
  );
}

export default function HomePage() {
  return (
    <>
      <ExampleSelector />
      <MainLayout
        editorPane={<EditorPane />}
        analysisPanel={<AnalysisPanel />}
      />
    </>
  );
}
