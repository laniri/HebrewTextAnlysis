import type { ReactNode } from 'react';

interface MainLayoutProps {
  editorPane: ReactNode;
  analysisPanel: ReactNode;
}

/**
 * Two-pane layout for the main writing coach view.
 *
 * In RTL mode:
 * - The "left" pane (editor) appears on the right side visually
 * - The "right" pane (analysis) appears on the left side visually
 *
 * On mobile (≤768px), panes stack vertically: editor on top, analysis below.
 */
export default function MainLayout({ editorPane, analysisPanel }: MainLayoutProps) {
  return (
    <div className="flex flex-1 flex-col md:flex-row min-h-0 p-5 gap-4" style={{ maxHeight: 'calc(100vh - 4rem)' }}>
      {/* Editor Pane — takes ~60% on desktop */}
      <section
        className="flex-1 min-h-[300px] md:min-h-0 md:w-3/5 overflow-hidden rounded-xl border"
        style={{
          borderColor: 'var(--border-light)',
          boxShadow: 'var(--shadow-sm)',
        }}
        aria-label="עורך טקסט"
      >
        <div className="h-full" style={{ backgroundColor: 'var(--bg-editor)' }}>
          {editorPane}
        </div>
      </section>

      {/* Analysis Panel — takes ~40% on desktop */}
      <aside
        className="md:w-2/5 overflow-y-auto rounded-xl border"
        style={{
          backgroundColor: 'var(--bg-primary)',
          borderColor: 'var(--border-light)',
          boxShadow: 'var(--shadow-sm)',
        }}
        aria-label="לוח ניתוח"
      >
        <div className="p-4 lg:p-6">
          {analysisPanel}
        </div>
      </aside>
    </div>
  );
}
