import { useEffect, useRef } from 'react';
import type { AnalyzeResponse } from '../types';

/**
 * Hook that applies Monaco editor decorations for sentence highlights
 * and cohesion gap underlines based on analysis results.
 *
 * Usage: call from MonacoEditor after the editor mounts.
 *   useInlineAnnotations(editorRef.current, analysisResult);
 *
 * Decoration rules:
 *  - Sentences with complexity > 0.7  → red background
 *  - Sentences with complexity 0.4–0.7 → yellow background
 *  - Cohesion gaps → red wavy underline between adjacent sentences
 *  - Hover tooltips show issue type, explanation, and suggested action in Hebrew
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type MonacoEditor = any;

const HIGHLIGHT_CLASSES: Record<string, string> = {
  red: 'sentence-highlight-red',
  yellow: 'sentence-highlight-yellow',
};

export function useInlineAnnotations(
  editor: MonacoEditor | null,
  analysisResult: AnalyzeResponse | null,
) {
  const decorationsRef = useRef<string[]>([]);

  useEffect(() => {
    if (!editor || !analysisResult) {
      // Clear decorations when there's no analysis
      if (editor && decorationsRef.current.length > 0) {
        decorationsRef.current = editor.deltaDecorations(
          decorationsRef.current,
          [],
        );
      }
      return;
    }

    const model = editor.getModel();
    if (!model) return;

    const newDecorations: {
      range: { startLineNumber: number; startColumn: number; endLineNumber: number; endColumn: number };
      options: Record<string, unknown>;
    }[] = [];

    // ── Sentence highlights ──
    for (const sentence of analysisResult.sentences) {
      const className = HIGHLIGHT_CLASSES[sentence.highlight];
      if (!className) continue; // "none" → no decoration

      const startPos = model.getPositionAt(sentence.char_start);
      const endPos = model.getPositionAt(sentence.char_end);

      const severityLabel =
        sentence.highlight === 'red' ? 'גבוהה' : 'בינונית';
      const hoverMessage = `**מורכבות ${severityLabel}** (${(sentence.complexity * 100).toFixed(0)}%)\nנסו לפשט את המשפט — פצלו למשפטים קצרים יותר.`;

      newDecorations.push({
        range: {
          startLineNumber: startPos.lineNumber,
          startColumn: startPos.column,
          endLineNumber: endPos.lineNumber,
          endColumn: endPos.column,
        },
        options: {
          className,
          hoverMessage: { value: hoverMessage },
          isWholeLine: false,
        },
      });
    }

    // ── Cohesion gap underlines ──
    for (const gap of analysisResult.cohesion_gaps) {
      const startPos = model.getPositionAt(gap.char_start);
      const endPos = model.getPositionAt(gap.char_end);

      const hoverMessage = `**פער קוהרנטיות** (חומרה: ${(gap.severity * 100).toFixed(0)}%)\nהוסיפו מילות קישור בין המשפטים כדי לשפר את הזרימה.`;

      newDecorations.push({
        range: {
          startLineNumber: startPos.lineNumber,
          startColumn: startPos.column,
          endLineNumber: endPos.lineNumber,
          endColumn: endPos.column,
        },
        options: {
          className: 'cohesion-gap-underline',
          hoverMessage: { value: hoverMessage },
          isWholeLine: false,
        },
      });
    }

    decorationsRef.current = editor.deltaDecorations(
      decorationsRef.current,
      newDecorations,
    );
  }, [editor, analysisResult]);
}
