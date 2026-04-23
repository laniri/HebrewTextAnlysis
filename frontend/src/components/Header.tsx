import { Link, useLocation } from 'react-router-dom';
import { useAppStore } from '../store/useAppStore';
import { exportAnalysisPdf } from '../utils/exportPdf';
import { encodeShareUrl } from '../utils/shareUrl';

const navLinks = [
  { to: '/', label: 'עורך' },
  { to: '/methodology', label: 'מתודולוגיה' },
  { to: '/model', label: 'המודל' },
];

export default function Header() {
  const location = useLocation();
  const { text, analysisResult } = useAppStore();

  const handleExport = () => {
    if (!analysisResult || !text.trim()) {
      alert('אין טקסט לייצוא — הקלידו טקסט וחכו לניתוח');
      return;
    }
    exportAnalysisPdf(text, analysisResult);
  };

  const handleShare = () => {
    if (!text.trim()) {
      alert('אין טקסט לשיתוף — הקלידו טקסט תחילה');
      return;
    }
    const url = encodeShareUrl(text);
    navigator.clipboard.writeText(url).then(() => {
      alert('הקישור הועתק ללוח!');
    }).catch(() => {
      window.prompt('קישור לשיתוף:', url);
    });
  };

  return (
    <header className="sticky top-0 z-50 border-b bg-white/80 backdrop-blur-md" style={{ borderColor: 'var(--border-light)' }}>
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Logo + Title */}
        <Link to="/" className="flex items-center gap-3 no-underline group">
          <div
            className="flex h-9 w-9 items-center justify-center rounded-lg text-white text-sm font-bold shadow-sm"
            style={{ background: 'linear-gradient(135deg, var(--primary-800), var(--accent-700))' }}
            aria-hidden="true"
          >
            עב
          </div>
          <span
            className="text-lg font-bold tracking-tight transition-colors"
            style={{
              fontFamily: 'var(--font-heading)',
              color: 'var(--text-primary)',
            }}
          >
            מאמן כתיבה בעברית
          </span>
        </Link>

        {/* Navigation */}
        <nav className="flex items-center gap-1" aria-label="ניווט ראשי">
          {navLinks.map((link) => {
            const isActive = location.pathname === link.to;
            return (
              <Link
                key={link.to}
                to={link.to}
                className="relative rounded-lg px-3 py-2 text-sm font-medium no-underline transition-all"
                style={{
                  color: isActive ? 'var(--accent-700)' : 'var(--text-secondary)',
                  backgroundColor: isActive ? 'var(--accent-50)' : 'transparent',
                }}
                onMouseEnter={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.color = 'var(--text-primary)';
                    e.currentTarget.style.backgroundColor = 'var(--primary-50)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.color = 'var(--text-secondary)';
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }
                }}
                aria-current={isActive ? 'page' : undefined}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <a
            href="https://github.com/laniri/HebrewTextAnlysis"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-center rounded-lg p-1.5 transition-all"
            style={{ color: 'var(--text-secondary)', backgroundColor: 'transparent' }}
            onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-primary)'; e.currentTarget.style.backgroundColor = 'var(--primary-50)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-secondary)'; e.currentTarget.style.backgroundColor = 'transparent'; }}
            aria-label="GitHub"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
            </svg>
          </a>
          <button
            onClick={handleExport}
            className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm font-medium transition-all cursor-pointer"
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
            aria-label="ייצוא ל-PDF"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M3 14h10M8 2v9m0 0L5 8m3 3l3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            ייצוא
          </button>

          <button
            onClick={handleShare}
            className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium text-white transition-all cursor-pointer"
            style={{
              background: 'linear-gradient(135deg, var(--accent-600), var(--accent-700))',
              boxShadow: 'var(--shadow-sm)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'linear-gradient(135deg, var(--accent-500), var(--accent-600))';
              e.currentTarget.style.boxShadow = 'var(--shadow-md)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'linear-gradient(135deg, var(--accent-600), var(--accent-700))';
              e.currentTarget.style.boxShadow = 'var(--shadow-sm)';
            }}
            aria-label="שיתוף קישור"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M6 10l4-4M6.5 4.5L5.1 5.9a2.5 2.5 0 000 3.536l.707.707a2.5 2.5 0 003.536 0L10.5 9M9.5 11.5l1.4-1.4a2.5 2.5 0 000-3.536l-.707-.707a2.5 2.5 0 00-3.536 0L5.5 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            שיתוף
          </button>
        </div>
      </div>
    </header>
  );
}
