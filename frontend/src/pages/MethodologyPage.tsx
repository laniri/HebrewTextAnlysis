import { useState } from 'react';

/* ── Collapsible Section Component ── */
function CollapsibleSection({
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
        style={{
          backgroundColor: isOpen ? 'var(--accent-50)' : 'transparent',
          border: 'none',
        }}
        aria-expanded={isOpen}
      >
        <div className="flex items-center gap-3">
          <span className="text-xl" aria-hidden="true">{icon}</span>
          <h2
            className="text-lg font-bold"
            style={{
              fontFamily: 'var(--font-heading)',
              color: 'var(--text-primary)',
            }}
          >
            {title}
          </h2>
        </div>
        <svg
          width="20"
          height="20"
          viewBox="0 0 20 20"
          fill="none"
          className="shrink-0 transition-transform"
          style={{
            transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'var(--transition-base)',
          }}
          aria-hidden="true"
        >
          <path
            d="M5 7.5L10 12.5L15 7.5"
            stroke="var(--text-muted)"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>

      <div
        style={{
          maxHeight: isOpen ? '5000px' : '0',
          opacity: isOpen ? 1 : 0,
          transition: 'max-height 0.4s ease, opacity 0.3s ease',
          overflow: 'hidden',
        }}
      >
        <div className="px-5 pb-5 pt-2">{children}</div>
      </div>
    </div>
  );
}

/* ── Feature Category Sub-section ── */
function FeatureCategory({
  title,
  color,
  features,
}: {
  title: string;
  color: string;
  features: { name: string; description: string }[];
}) {
  return (
    <div className="mb-5 last:mb-0">
      <h3
        className="mb-2.5 flex items-center gap-2 text-sm font-bold"
        style={{ color }}
      >
        <span
          className="inline-block h-2.5 w-2.5 rounded-sm"
          style={{ backgroundColor: color }}
        />
        {title}
      </h3>
      <div className="grid gap-2 sm:grid-cols-2">
        {features.map((f) => (
          <div
            key={f.name}
            className="rounded-lg px-3 py-2"
            style={{
              backgroundColor: 'var(--warm-50)',
              borderRight: `2px solid ${color}`,
            }}
          >
            <span
              className="block text-xs font-semibold"
              style={{ color: 'var(--text-primary)' }}
            >
              {f.name}
            </span>
            <span
              className="block text-xs leading-relaxed"
              style={{ color: 'var(--text-muted)' }}
            >
              {f.description}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Data: Features ── */
const MORPHOLOGY_FEATURES = [
  { name: 'יחס פעלים', description: 'שיעור המילים המתויגות כפועל מתוך כלל המילים' },
  { name: 'התפלגות בניינים', description: 'היסטוגרמה של דפוסי הטיית הפועל (בניינים) בעברית' },
  { name: 'צפיפות תחיליות', description: 'מספר ממוצע של תחיליות למילה (ו, ב, ל וכו\')' },
  { name: 'יחס כינויי סיומת', description: 'שיעור המילים עם כינוי גוף בסיומת' },
  { name: 'עמימות מורפולוגית', description: 'מספר ממוצע של ניתוחים מורפולוגיים אפשריים למילה' },
  { name: 'שיעור שגיאות התאמה', description: 'שיעור זוגות נושא-נשוא/תואר עם אי-התאמה במין או מספר' },
  { name: 'אנטרופיית בניינים', description: 'מדד שאנון על התפלגות הבניינים — ערך גבוה = שימוש מגוון יותר בפעלים' },
  { name: 'יחס סמיכות', description: 'שיעור זוגות שם-עצם סמוכים (צירופי סמיכות)' },
];

const SYNTAX_FEATURES = [
  { name: 'אורך משפט ממוצע', description: 'מספר מילים ממוצע למשפט' },
  { name: 'עומק עץ ממוצע', description: 'עומק מרבי ממוצע של עצי תלות' },
  { name: 'עומק עץ מרבי', description: 'עומק עץ התלות המרבי בכל המשפטים' },
  { name: 'מרחק תלות ממוצע', description: 'מרחק מוחלט ממוצע בין מילים לראשי התלות שלהן' },
  { name: 'פסוקיות למשפט', description: 'מספר יחסי תלות של פסוקיות משנה למשפט' },
  { name: 'יחס הסתעפות ימנית', description: 'שיעור תלויות שבהן המילה עוקבת לראש שלה' },
  { name: 'שונות מרחק תלות', description: 'שונות מדגמית של מרחקי תלות' },
  { name: 'אנטרופיית סוגי פסוקיות', description: 'מדד שאנון על סוגי יחסי התלות — ערך גבוה = מבנים תחביריים מגוונים יותר' },
];

const LEXICON_FEATURES = [
  { name: 'יחס סוג-אסימון', description: 'צורות ייחודיות חלקי סך המילים' },
  { name: 'יחס הפקס', description: 'מילים המופיעות פעם אחת בלבד חלקי סך המילים' },
  { name: 'אורך מילה ממוצע', description: 'מספר תווים ממוצע למילה' },
  { name: 'מגוון למות', description: 'למות ייחודיות חלקי סך המילים' },
  { name: 'יחס מילים נדירות', description: 'מילים עם תדירות קורפוס נמוכה חלקי סך המילים' },
  { name: 'יחס מילות תוכן', description: 'שמות עצם, פעלים, תארים ותוארי פועל חלקי סך המילים' },
];

const STRUCTURE_FEATURES = [
  { name: 'שונות אורך משפטים', description: 'שונות מדגמית של מספר המילים בין משפטים' },
  { name: 'יחס משפטים ארוכים', description: 'משפטים מעל סף (ברירת מחדל 20) חלקי סך המשפטים' },
  { name: 'יחס פיסוק', description: 'סימני פיסוק חלקי סך המילים' },
  { name: 'יחס משפטים קצרים', description: 'משפטים עם פחות מ-3 מילים חלקי סך המשפטים' },
  { name: 'יחס פיסוק סופי חסר', description: 'משפטים שאינם מסתיימים בנקודה, סימן קריאה או שאלה' },
];

const DISCOURSE_FEATURES = [
  { name: 'יחס מילות קישור', description: 'מילות קישור בעברית חלקי סך המשפטים' },
  { name: 'חפיפה בין משפטים', description: 'דמיון ז\'אקאר ממוצע של קבוצות למות בין משפטים סמוכים' },
  { name: 'יחס כינויים לשמות', description: 'יחס כינויי גוף לשמות עצם — ערך גבוה = עמימות אפשרית' },
];

const STYLE_FEATURES = [
  { name: 'מגמת אורך משפטים', description: 'שיפוע רגרסיה ליניארית על אורכי משפטים — חיובי = משפטים מתארכים' },
  { name: 'שונות התפלגות חלקי דיבור', description: 'ממוצע שונויות של היסטוגרמות חלקי דיבור בין משפטים' },
];

/* ── Data: Scores ── */
const SCORES_DATA = [
  {
    name: 'קושי',
    key: 'difficulty',
    range: '0–1',
    description: 'כמה קשה הטקסט לקריאה. ערך גבוה = קשה יותר.',
    formula: 'סכום משוקלל של אורך משפט ממוצע (0.30), עומק עץ ממוצע (0.25), יחס הפקס (0.25), ועמימות מורפולוגית (0.20). תכונות חסרות מוחרגות והמשקלות מנורמלים מחדש.',
  },
  {
    name: 'סגנון',
    key: 'style',
    range: 'בדרך כלל −0.15 עד 0.20',
    description: 'רישום סגנוני ועקביות. ערך גבוה = סגנון מובהק יותר.',
    formula: 'יחס כינויי סיומת (+0.25) − יחס הפקס (0.25) + |מגמת אורך משפטים| (+0.20) − שונות התפלגות חלקי דיבור (0.15) + יחס כינויים לשמות (+0.15).',
  },
  {
    name: 'שטף',
    key: 'fluency',
    range: '0–1',
    description: 'עקביות מבנית וסדירות. ערך גבוה = שוטף יותר.',
    formula: 'ממוצע של יחס פיסוק, שונות אורך משפטים הפוכה, ושונות התפלגות חלקי דיבור הפוכה. שונות נמוכה = שטף גבוה.',
  },
  {
    name: 'קוהרנטיות',
    key: 'cohesion',
    range: '0–1',
    description: 'כמה טוב הטקסט שומר על קשר בין משפטים. ערך גבוה = קוהרנטי יותר.',
    formula: 'שילוב משוקלל — 0.4 × יחס מילות קישור + 0.3 × חפיפה בין משפטים + 0.3 × (1 − יחס כינויים לשמות). יחס כינויים הפוך מודד בהירות הפניה.',
  },
  {
    name: 'מורכבות',
    key: 'complexity',
    range: '0–1',
    description: 'עושר מורפו-תחבירי — מגוון מבנים דקדוקיים, ללא תלות בקושי קריאה.',
    formula: 'ממוצע של אנטרופיית בניינים, שיעור שגיאות התאמה, שונות התפלגות חלקי דיבור, אנטרופיית סוגי פסוקיות, ויחס סמיכות (×0.5).',
  },
];

/* ── Data: Issues (17 types) ── */
const ISSUES_DATA = [
  { group: 'מורפולוגיה', issues: [
    { name: 'שגיאות התאמה', formula: 'ציון רך של שיעור שגיאות התאמה' },
    { name: 'עמימות מורפולוגית', formula: 'ציון רך של עמימות מורפולוגית' },
    { name: 'מגוון מורפולוגי נמוך', formula: '1 − ציון רך של אנטרופיית בניינים' },
  ]},
  { group: 'תחביר', issues: [
    { name: 'מורכבות משפט', formula: '0.6 × ציון רך(מספר מילים) + 0.4 × ציון רך(עומק עץ) — לכל משפט' },
    { name: 'פיזור תלויות', formula: 'ציון רך של שונות מרחק תלות' },
    { name: 'הסתעפות מוגזמת', formula: 'ציון רך של יחס הסתעפות ימנית' },
  ]},
  { group: 'לקסיקון', issues: [
    { name: 'מגוון לקסיקלי נמוך', formula: '0.6 × (1−ציון רך(מגוון למות)) + 0.4 × (1−ציון רך(יחס סוג-אסימון))' },
    { name: 'שימוש יתר במילים נדירות', formula: 'ציון רך של יחס מילים נדירות' },
    { name: 'צפיפות תוכן נמוכה', formula: '1 − ציון רך של יחס מילות תוכן' },
  ]},
  { group: 'מבנה', issues: [
    { name: 'שונות אורך משפטים', formula: 'ציון רך של שונות אורך משפטים' },
    { name: 'בעיות פיסוק', formula: '0.5 × (1−ציון רך(יחס פיסוק)) + 0.5 × ציון רך(פיסוק סופי חסר)' },
    { name: 'פיצול', formula: 'ציון רך של יחס משפטים קצרים' },
  ]},
  { group: 'שיח', issues: [
    { name: 'קוהרנטיות חלשה', formula: '1 − ציון רך(דמיון) — לכל זוג משפטים סמוכים' },
    { name: 'מילות קישור חסרות', formula: '1 − ציון רך של יחס מילות קישור' },
    { name: 'עמימות כינויים', formula: 'ציון רך של יחס כינויים לשמות' },
  ]},
  { group: 'סגנון', issues: [
    { name: 'חוסר עקביות מבנית', formula: 'ציון רך של שונות התפלגות חלקי דיבור' },
    { name: 'סחיפת מגמת משפטים', formula: '|ציון רך(מגמת אורך) − 0.5| × 2' },
  ]},
];

/* ── Data: Diagnoses (8 rules) ── */
const DIAGNOSES_DATA = [
  { name: 'אוצר מילים מצומצם', type: 'low_lexical_diversity', formula: '0.7 × חומרת "מגוון לקסיקלי נמוך" + 0.3 × חומרת "צפיפות תוכן נמוכה"', threshold: '0.6' },
  { name: 'שימוש יתר בכינויי גוף', type: 'pronoun_overuse', formula: '0.8 × חומרת "עמימות כינויים" + 0.2 × ציון קוהרנטיות', threshold: '0.6' },
  { name: 'קוהרנטיות נמוכה', type: 'low_cohesion', formula: '0.6 × חומרת "קוהרנטיות חלשה" + 0.4 × חומרת "מילות קישור חסרות"', threshold: '0.6' },
  { name: 'משפטים מורכבים מדי', type: 'sentence_over_complexity', formula: '0.7 × ממוצע חומרת "מורכבות משפט" + 0.3 × ציון קושי', threshold: '0.65' },
  { name: 'חוסר עקביות מבנית', type: 'structural_inconsistency', formula: '0.6 × חומרת "חוסר עקביות מבנית" + 0.4 × ציון שטף', threshold: '0.6' },
  { name: 'מגוון מורפולוגי נמוך', type: 'low_morphological_richness', formula: '0.7 × חומרת "מגוון מורפולוגי נמוך" + 0.3 × ציון מורכבות', threshold: '0.6' },
  { name: 'כתיבה מקוטעת', type: 'fragmented_writing', formula: 'חומרת "פיצול" ישירות', threshold: '0.6' },
  { name: 'בעיות פיסוק', type: 'punctuation_deficiency', formula: 'חומרת "בעיות פיסוק" ישירות', threshold: '0.6' },
];

/* ── Data: Interventions (4 types) ── */
const INTERVENTIONS_DATA = [
  {
    name: 'הרחבת אוצר מילים',
    type: 'vocabulary_expansion',
    triggers: ['אוצר מילים מצומצם', 'מגוון מורפולוגי נמוך'],
    description: 'מתמקדת בחולשה לקסיקלית ומורפולוגית. כוללת תרגילי החלפת מילים נרדפות, שימוש בצורות מילים מגוונות, וניסוח מחדש.',
  },
  {
    name: 'הבהרת כינויי גוף',
    type: 'pronoun_clarification',
    triggers: ['שימוש יתר בכינויי גוף'],
    description: 'מתמקדת בעמימות הפניה מכינויי גוף. כוללת החלפת כינויים עמומים בשמות עצם מפורשים והפחתת צפיפות כינויים.',
  },
  {
    name: 'פישוט משפטים',
    type: 'sentence_simplification',
    triggers: ['משפטים מורכבים מדי', 'חוסר עקביות מבנית', 'כתיבה מקוטעת'],
    description: 'מתמקדת במבני משפטים מורכבים, לא עקביים או מקוטעים. כוללת פירוק משפטים ארוכים, הפחתת קינון פסוקיות, ואיחוד קטעים.',
  },
  {
    name: 'שיפור קוהרנטיות',
    type: 'cohesion_improvement',
    triggers: ['קוהרנטיות נמוכה', 'בעיות פיסוק'],
    description: 'מתמקדת בקישוריות שיח חלשה ובעיות פיסוק. כוללת הוספת מילות קישור, הגברת חפיפה לקסיקלית, ותיקון פיסוק.',
  },
];

/* ── Main Component ── */
export default function MethodologyPage() {
  return (
    <div className="mx-auto max-w-3xl flex-1 px-4 py-10 sm:px-6 lg:px-8">
      {/* Page Header */}
      <div className="mb-10 text-center">
        <div
          className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl"
          style={{ backgroundColor: 'var(--accent-50)' }}
        >
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path
              d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25"
              stroke="var(--accent-600)"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
        <h1
          className="mb-3 text-3xl"
          style={{
            fontFamily: 'var(--font-heading)',
            color: 'var(--text-primary)',
          }}
        >
          המתודולוגיה שלנו
        </h1>
        <p
          className="mx-auto max-w-lg text-base leading-relaxed"
          style={{ color: 'var(--text-secondary)' }}
        >
          כיצד המערכת מנתחת את הטקסט שלכם ומספקת משוב לשוני מדויק — מחילוץ תכונות ועד המלצות מותאמות אישית
        </p>
      </div>

      {/* Collapsible Sections */}
      <div className="space-y-4">
        {/* ── Section 1: Features ── */}
        <CollapsibleSection title="תכונות" icon="🔬" defaultOpen>
          <p
            className="mb-4 text-sm leading-relaxed"
            style={{ color: 'var(--text-secondary)' }}
          >
            המערכת מחלצת למעלה מ-30 תכונות לשוניות מהטקסט, מחולקות לשש קטגוריות. כל תכונה מודדת היבט שונה של הכתיבה.
          </p>

          <FeatureCategory
            title="מורפולוגיה"
            color="var(--accent-700)"
            features={MORPHOLOGY_FEATURES}
          />
          <FeatureCategory
            title="תחביר"
            color="var(--primary-600)"
            features={SYNTAX_FEATURES}
          />
          <FeatureCategory
            title="לקסיקון"
            color="#8b5cf6"
            features={LEXICON_FEATURES}
          />
          <FeatureCategory
            title="מבנה"
            color="#d97706"
            features={STRUCTURE_FEATURES}
          />
          <FeatureCategory
            title="שיח"
            color="#dc2626"
            features={DISCOURSE_FEATURES}
          />
          <FeatureCategory
            title="סגנון"
            color="#0891b2"
            features={STYLE_FEATURES}
          />
        </CollapsibleSection>

        {/* ── Section 2: Scores ── */}
        <CollapsibleSection title="ציונים" icon="📊">
          <p
            className="mb-4 text-sm leading-relaxed"
            style={{ color: 'var(--text-secondary)' }}
          >
            חמישה ציונים מורכבים מחושבים מהתכונות. כל ציון הוא שילוב משוקלל של ערכי תכונות מנורמלים. הציונים בלתי תלויים זה בזה.
          </p>

          <div className="space-y-3">
            {SCORES_DATA.map((score) => (
              <div
                key={score.key}
                className="rounded-lg border p-4"
                style={{
                  borderColor: 'var(--border-light)',
                  backgroundColor: 'var(--warm-50)',
                }}
              >
                <div className="mb-2 flex items-center justify-between">
                  <h4
                    className="text-sm font-bold"
                    style={{
                      fontFamily: 'var(--font-heading)',
                      color: 'var(--text-primary)',
                    }}
                  >
                    {score.name}
                  </h4>
                  <span
                    className="rounded-full px-2 py-0.5 text-xs font-medium"
                    style={{
                      backgroundColor: 'var(--accent-100)',
                      color: 'var(--accent-800)',
                    }}
                  >
                    {score.range}
                  </span>
                </div>
                <p
                  className="mb-2 text-sm"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  {score.description}
                </p>
                <div
                  className="rounded-md px-3 py-2 text-xs leading-relaxed"
                  style={{
                    backgroundColor: 'var(--bg-secondary)',
                    color: 'var(--text-muted)',
                    borderRight: '2px solid var(--accent-300)',
                  }}
                >
                  <span className="font-semibold" style={{ color: 'var(--text-secondary)' }}>
                    נוסחה:{' '}
                  </span>
                  {score.formula}
                </div>
              </div>
            ))}
          </div>
        </CollapsibleSection>

        {/* ── Section 3: Issues ── */}
        <CollapsibleSection title="בעיות" icon="⚠️">
          <p
            className="mb-4 text-sm leading-relaxed"
            style={{ color: 'var(--text-secondary)' }}
          >
            17 סוגי בעיות ב-6 קבוצות. כל בעיה מזוהה באמצעות ציון רך (sigmoid) המבוסס על סטיית ערך התכונה מממוצע הקורפוס.
          </p>

          <div className="space-y-4">
            {ISSUES_DATA.map((group) => (
              <div key={group.group}>
                <h4
                  className="mb-2 text-sm font-bold"
                  style={{ color: 'var(--primary-700)' }}
                >
                  {group.group}
                </h4>
                <div className="space-y-2">
                  {group.issues.map((issue) => (
                    <div
                      key={issue.name}
                      className="flex flex-col gap-1 rounded-lg px-3 py-2 sm:flex-row sm:items-start sm:justify-between sm:gap-4"
                      style={{ backgroundColor: 'var(--warm-50)' }}
                    >
                      <span
                        className="text-sm font-medium"
                        style={{ color: 'var(--text-primary)' }}
                      >
                        {issue.name}
                      </span>
                      <span
                        className="text-xs leading-relaxed"
                        style={{ color: 'var(--text-muted)' }}
                      >
                        {issue.formula}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </CollapsibleSection>

        {/* ── Section 4: Diagnoses ── */}
        <CollapsibleSection title="אבחנות" icon="🩺">
          <p
            className="mb-4 text-sm leading-relaxed"
            style={{ color: 'var(--text-secondary)' }}
          >
            8 כללי אבחון מצרפים דפוסי בעיות וציונים מורכבים לאבחנות לשוניות משמעותיות. כל אבחנה מופעלת רק כאשר חומרתה עוברת את הסף.
          </p>

          <div className="space-y-2">
            {DIAGNOSES_DATA.map((diag) => (
              <div
                key={diag.type}
                className="rounded-lg border p-3"
                style={{
                  borderColor: 'var(--border-light)',
                  backgroundColor: 'var(--warm-50)',
                }}
              >
                <div className="mb-1.5 flex items-center justify-between">
                  <h4
                    className="text-sm font-bold"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    {diag.name}
                  </h4>
                  <span
                    className="rounded-full px-2 py-0.5 text-xs font-medium"
                    style={{
                      backgroundColor: 'var(--severity-medium)',
                      color: '#fff',
                    }}
                  >
                    סף: {diag.threshold}
                  </span>
                </div>
                <p
                  className="text-xs leading-relaxed"
                  style={{ color: 'var(--text-muted)' }}
                >
                  {diag.formula}
                </p>
              </div>
            ))}
          </div>
        </CollapsibleSection>

        {/* ── Section 5: Interventions ── */}
        <CollapsibleSection title="המלצות" icon="💡">
          <p
            className="mb-4 text-sm leading-relaxed"
            style={{ color: 'var(--text-secondary)' }}
          >
            4 סוגי התערבות פדגוגית ממופים מהאבחנות. כל אבחנה ממופה לסוג התערבות אחד בדיוק.
          </p>

          <div className="space-y-3">
            {INTERVENTIONS_DATA.map((intervention) => (
              <div
                key={intervention.type}
                className="rounded-lg border p-4"
                style={{
                  borderColor: 'var(--border-light)',
                  backgroundColor: 'var(--warm-50)',
                }}
              >
                <h4
                  className="mb-1.5 text-sm font-bold"
                  style={{
                    fontFamily: 'var(--font-heading)',
                    color: 'var(--accent-800)',
                  }}
                >
                  {intervention.name}
                </h4>
                <p
                  className="mb-2 text-sm leading-relaxed"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  {intervention.description}
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {intervention.triggers.map((trigger) => (
                    <span
                      key={trigger}
                      className="rounded-full px-2.5 py-0.5 text-xs font-medium"
                      style={{
                        backgroundColor: 'var(--primary-100)',
                        color: 'var(--primary-700)',
                      }}
                    >
                      {trigger}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </CollapsibleSection>
      </div>

      {/* Footer note */}
      <div className="mt-8 text-center">
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          המתודולוגיה מבוססת על קורפוס ויקיפדיה עברית ומודל DictaBERT מזוקק
        </p>
      </div>
    </div>
  );
}
