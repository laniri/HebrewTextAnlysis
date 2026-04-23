import Editor, { type OnMount } from '@monaco-editor/react';
import { useRef, useCallback, useEffect } from 'react';
import { useAppStore } from '../store/useAppStore';

function Placeholder() {
  const { text } = useAppStore();
  if (text.trim()) return null;

  return (
    <div
      style={{
        position: 'absolute',
        top: 20,
        right: 20,
        left: 20,
        pointerEvents: 'none',
        zIndex: 1,
        direction: 'rtl',
        textAlign: 'right',
        fontFamily: "'Heebo', system-ui, sans-serif",
        fontSize: 16,
        lineHeight: 1.7,
        color: '#829ab1',
        opacity: 0.85,
      }}
    >
      הקלידו או הדביקו טקסט בעברית כאן, או בחרו דוגמה מהרשימה למעלה ↑
    </div>
  );
}

export default function MonacoEditor() {
  const { text, setText, analyzeText } = useAppStore();
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const editorRef = useRef<any>(null);

  const handleChange = useCallback(
    (value: string | undefined) => {
      const newText = value ?? '';
      setText(newText);

      if (timerRef.current) clearTimeout(timerRef.current);
      if (newText.trim()) {
        timerRef.current = setTimeout(() => analyzeText(newText), 800);
      }
    },
    [setText, analyzeText],
  );

  const handleMount: OnMount = useCallback((editor) => {
    editorRef.current = editor;
  }, []);

  const handleContainerClick = useCallback(() => {
    if (editorRef.current) {
      editorRef.current.focus();
    }
  }, []);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  return (
    <div className="h-full" onClick={handleContainerClick} dir="ltr" style={{ position: 'relative' }}>
      <Placeholder />
      <Editor
        height="100%"
        language="plaintext"
        theme="vs"
        value={text}
        onChange={handleChange}
        onMount={handleMount}
        options={{
          wordWrap: 'on',
          minimap: { enabled: false },
          lineNumbers: 'off',
          renderWhitespace: 'none',
          fontSize: 16,
          fontFamily: "'Heebo', system-ui, sans-serif",
          padding: { top: 16, bottom: 16 },
          scrollBeyondLastLine: false,
          automaticLayout: true,
          cursorStyle: 'line',
          renderLineHighlight: 'none',
        }}
      />
    </div>
  );
}
