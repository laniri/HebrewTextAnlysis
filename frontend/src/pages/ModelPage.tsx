import { useState } from 'react';

/* ── Collapsible Section ── */
function Section({
  title,
  icon,
  children,
  defaultOpen = false,
}: {
  title: string;
  icon: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  return (
    <div
      className="overflow-hidden rounded-xl border transition-all"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        borderColor: isOpen ? 'var(--accent-300)' : 'var(--border-light)',
        boxShadow: isOpen ? 'var(--shadow-md)' : 'var(--shadow-sm)',
      }}
    >
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full cursor-pointer items-center justify-between px-5 py-4 text-right transition-colors"
        style={{ backgroundColor: isOpen ? 'var(--accent-50)' : 'transparent', border: 'none' }}
        aria-expanded={isOpen}
      >
        <div className="flex items-center gap-3">
          <span className="text-xl" aria-hidden="true">{icon}</span>
          <h2 className="text-lg font-bold" style={{ fontFamily: 'var(--font-heading)', color: 'var(--text-primary)' }}>
            {title}
          </h2>
        </div>
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" className="shrink-0" style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'var(--transition-base)' }} aria-hidden="true">
          <path d="M5 7.5L10 12.5L15 7.5" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      <div style={{ maxHeight: isOpen ? '5000px' : '0', opacity: isOpen ? 1 : 0, transition: 'max-height 0.4s ease, opacity 0.3s ease', overflow: 'hidden' }}>
        <div className="px-5 pb-5 pt-2">{children}</div>
      </div>
    </div>
  );
}

/* ── Metric Bar ── */
function MetricBar({ label, value, max = 1, format = 'pct' }: { label: string; value: number; max?: number; format?: 'pct' | 'rmse' }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = format === 'rmse'
    ? (value < 0.1 ? 'var(--severity-low)' : value < 0.15 ? 'var(--severity-medium)' : 'var(--severity-high)')
    : (value > 0.7 ? 'var(--severity-low)' : value > 0.4 ? 'var(--severity-medium)' : 'var(--severity-high)');
  const display = format === 'rmse' ? value.toFixed(3) : `${(value * 100).toFixed(0)}%`;

  return (
    <div className="flex items-center gap-3 py-1.5">
      <span className="w-40 shrink-0 text-sm" style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--warm-200)' }}>
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color, transition: 'width 0.5s ease' }} />
      </div>
      <span className="w-14 text-left text-sm font-bold" style={{ color: 'var(--text-primary)' }}>{display}</span>
    </div>
  );
}

/* ── Score RMSE data ── */
const SCORE_RMSE = [
  { label: 'קושי (difficulty)', value: 0.1055 },
  { label: 'סגנון (style)', value: 0.0809 },
  { label: 'שטף (fluency)', value: 0.1211 },
  { label: 'קוהרנטיות (cohesion)', value: 0.0987 },
  { label: 'מורכבות (complexity)', value: 0.0932 },
];

/* ── Issue F1 data ── */
const ISSUE_F1 = [
  { label: 'עמימות כינויים', value: 0.904 },
  { label: 'עמימות מורפולוגית', value: 0.883 },
  { label: 'צפיפות תוכן נמוכה', value: 0.862 },
  { label: 'מגוון לקסיקלי נמוך', value: 0.835 },
  { label: 'מילות קישור חסרות', value: 0.814 },
  { label: 'שימוש יתר במילים נדירות', value: 0.803 },
  { label: 'בעיות פיסוק', value: 0.775 },
  { label: 'מגוון מורפולוגי נמוך', value: 0.706 },
  { label: 'חוסר עקביות מבנית', value: 0.655 },
  { label: 'הסתעפות מוגזמת', value: 0.604 },
  { label: 'שגיאות התאמה', value: 0.543 },
  { label: 'שונות אורך משפטים', value: 0.457 },
  { label: 'פיזור תלויות', value: 0.405 },
  { label: 'פיצול', value: 0.381 },
  { label: 'סחיפת מגמת משפטים', value: 0.255 },
];

/* ── Diagnosis F1 data ── */
const DIAG_F1 = [
  { label: 'שימוש יתר בכינויי גוף', value: 0.630 },
  { label: 'אוצר מילים מצומצם', value: 0.597 },
  { label: 'מגוון מורפולוגי נמוך', value: 0.456 },
  { label: 'בעיות פיסוק', value: 0.304 },
  { label: 'חוסר עקביות מבנית', value: 0.154 },
];

export default function ModelPage() {
  return (
    <div className="mx-auto max-w-3xl flex-1 px-4 py-10 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="mb-10 text-center">
        <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl" style={{ backgroundColor: 'var(--primary-50)' }}>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" stroke="var(--primary-600)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <h1 className="mb-3 text-3xl" style={{ fontFamily: 'var(--font-heading)', color: 'var(--text-primary)' }}>
          על המודל
        </h1>
        <p className="mx-auto max-w-lg text-base leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
          כיצד המודל נבנה, אומן ומוערך — מארכיטקטורה ועד ביצועים
        </p>
      </div>

      <div className="space-y-4">
        {/* Architecture */}
        <Section title="ארכיטקטורה" icon="🏗️" defaultOpen>
          <div className="space-y-4 text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
            <p>
              המודל מבוסס על <strong style={{ color: 'var(--text-primary)' }}>DictaBERT</strong> — מודל שפה עברי מבית
              המכון הישראלי לטכנולוגיה (Technion) ו-DICTA. DictaBERT הוא מודל BERT שאומן מראש על קורפוס עברי גדול
              וכולל הבנה עמוקה של מורפולוגיה, תחביר ולקסיקון עבריים.
            </p>
            <div className="rounded-lg px-4 py-3" style={{ backgroundColor: 'var(--warm-50)', borderRight: '3px solid var(--primary-400)' }}>
              <p className="font-semibold" style={{ color: 'var(--text-primary)' }}>מבנה המודל</p>
              <ul className="mt-2 space-y-1">
                <li>• <strong>מקודד (Encoder):</strong> DictaBERT — 12 שכבות, 768 מימדים, 110M פרמטרים</li>
                <li>• <strong>ראש ציונים:</strong> Linear(768, 5) → sigmoid — 5 ציונים מורכבים</li>
                <li>• <strong>ראש בעיות:</strong> Linear(768, 17) → sigmoid — 17 סוגי בעיות</li>
                <li>• <strong>ראש אבחנות:</strong> Linear(768, 8) → sigmoid — 8 סוגי אבחנות</li>
                <li>• <strong>ראש משפטים:</strong> Linear(768, 1) → sigmoid — מורכבות לכל משפט</li>
                <li>• <strong>ראש זוגות:</strong> Linear(1536, 1) → sigmoid — קוהרנטיות בין משפטים סמוכים</li>
              </ul>
            </div>
            <p>
              העברה קדימה (forward pass) אחת מייצרת את כל התחזיות: ייצוג ה-CLS מזין את שלושת הראשים ברמת המסמך,
              ייצוגי המשפטים (mean pooling) מזינים את ראש המשפטים, ושרשור זוגות סמוכים מזין את ראש הזוגות.
            </p>
          </div>
        </Section>

        {/* Training */}
        <Section title="תהליך האימון" icon="🎓">
          <div className="space-y-4 text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
            <p>
              המודל אומן בגישת <strong style={{ color: 'var(--text-primary)' }}>זיקוק ידע (Knowledge Distillation)</strong> —
              צינור NLP דטרמיניסטי מלא (Stanza + YAP + חילוץ תכונות + ניקוד) משמש כ"מורה",
              והמודל לומד לשחזר את התוצאות שלו מטקסט גולמי בלבד.
            </p>
            <div className="rounded-lg px-4 py-3" style={{ backgroundColor: 'var(--warm-50)', borderRight: '3px solid var(--accent-400)' }}>
              <p className="font-semibold" style={{ color: 'var(--text-primary)' }}>פרמטרי אימון</p>
              <div className="mt-2 grid grid-cols-2 gap-2">
                <div>• <strong>Batch size:</strong> 16</div>
                <div>• <strong>Epochs:</strong> 6</div>
                <div>• <strong>Encoder LR:</strong> 2e-5</div>
                <div>• <strong>Heads LR:</strong> 1e-3</div>
                <div>• <strong>Sentence heads LR:</strong> 5e-3</div>
                <div>• <strong>Warmup:</strong> 10%</div>
                <div>• <strong>Max sequence:</strong> 512 tokens</div>
                <div>• <strong>Validation split:</strong> 10%</div>
              </div>
            </div>
            <p>
              פונקציית ההפסד היא שילוב רב-משימתי: MSE לציונים (משקל 1.0), BCE לבעיות (1.5), ו-BCE לאבחנות (2.0).
              שיעורי למידה דיפרנציאליים מאפשרים כוונון עדין של המקודד תוך אימון מהיר יותר של ראשי החיזוי.
            </p>
            <div className="rounded-lg px-4 py-3" style={{ backgroundColor: 'var(--primary-50)', borderRight: '3px solid var(--primary-400)' }}>
              <p className="font-semibold" style={{ color: 'var(--text-primary)' }}>נתוני אימון</p>
              <p className="mt-1">
                הקורפוס כולל טקסטים מוויקיפדיה העברית וקורפוס HeDC4 (Common Crawl) — חדשות, משפטים, בלוגים, פורומים ומסמכים ממשלתיים.
                כל טקסט עבר את צינור הניתוח המלא (Stanza + YAP) ליצירת תוויות רכות (soft labels) לאימון.
              </p>
            </div>
          </div>
        </Section>

        {/* Performance — Scores */}
        <Section title="ביצועים — ציונים" icon="📊">
          <p className="mb-4 text-sm" style={{ color: 'var(--text-secondary)' }}>
            RMSE (שורש שגיאה ריבועית ממוצעת) בין תחזיות המודל לתוויות הצינור. ערך נמוך = דיוק גבוה יותר.
          </p>
          <div className="space-y-1">
            {SCORE_RMSE.map((s) => (
              <MetricBar key={s.label} label={s.label} value={s.value} max={0.2} format="rmse" />
            ))}
          </div>
          <p className="mt-3 text-xs" style={{ color: 'var(--text-muted)' }}>
            ממוצע RMSE: 0.100 — המודל מדויק ברמה של ±10% מהציון האמיתי.
          </p>
        </Section>

        {/* Performance — Issues */}
        <Section title="ביצועים — בעיות" icon="⚠️">
          <p className="mb-4 text-sm" style={{ color: 'var(--text-secondary)' }}>
            F1 Score לכל סוג בעיה (סף 0.5). ערך גבוה = זיהוי מדויק יותר.
            Spearman rank correlation: <strong style={{ color: 'var(--text-primary)' }}>0.734</strong>
          </p>
          <div className="space-y-1">
            {ISSUE_F1.map((s) => (
              <MetricBar key={s.label} label={s.label} value={s.value} />
            ))}
          </div>
        </Section>

        {/* Performance — Diagnoses */}
        <Section title="ביצועים — אבחנות" icon="🩺">
          <p className="mb-4 text-sm" style={{ color: 'var(--text-secondary)' }}>
            F1 Score לכל סוג אבחנה (סף 0.5). אבחנות עם F1=0 לא הופיעו מספיק בנתוני הבדיקה.
            Spearman rank correlation: <strong style={{ color: 'var(--text-primary)' }}>0.344</strong>
          </p>
          <div className="space-y-1">
            {DIAG_F1.map((s) => (
              <MetricBar key={s.label} label={s.label} value={s.value} />
            ))}
          </div>
          <p className="mt-3 text-xs" style={{ color: 'var(--text-muted)' }}>
            אבחנות כמו "קוהרנטיות נמוכה" ו"משפטים מורכבים מדי" דורשות הרחבת נתוני אימון לשיפור הדיוק.
          </p>
        </Section>

        {/* Speed */}
        <Section title="מהירות" icon="⚡">
          <div className="space-y-3 text-sm" style={{ color: 'var(--text-secondary)' }}>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg p-4 text-center" style={{ backgroundColor: 'var(--accent-50)' }}>
                <div className="text-2xl font-bold" style={{ color: 'var(--accent-700)' }}>~50ms</div>
                <div className="text-xs" style={{ color: 'var(--text-muted)' }}>GPU (CUDA)</div>
              </div>
              <div className="rounded-lg p-4 text-center" style={{ backgroundColor: 'var(--primary-50)' }}>
                <div className="text-2xl font-bold" style={{ color: 'var(--primary-700)' }}>~200ms</div>
                <div className="text-xs" style={{ color: 'var(--text-muted)' }}>CPU</div>
              </div>
            </div>
            <p>
              לעומת הצינור המלא (Stanza + YAP) שדורש ~35 שניות לטקסט, המודל מספק תחזיות בזמן אמת
              מהעברה קדימה אחת — שיפור של פי 175–700 במהירות.
            </p>
          </div>
        </Section>
      </div>

      <div className="mt-8 text-center">
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          המודל מבוסס על DictaBERT מבית DICTA — המכון הישראלי לטכנולוגיה
        </p>
      </div>
    </div>
  );
}
