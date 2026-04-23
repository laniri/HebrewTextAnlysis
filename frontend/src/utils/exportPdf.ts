import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import type { AnalyzeResponse } from '../types';

const SCORE_LABELS: Record<string, string> = {
  difficulty: 'קושי',
  style: 'סגנון',
  fluency: 'שטף',
  cohesion: 'קוהרנטיות',
  complexity: 'מורכבות',
};

/**
 * Generate a PDF report by rendering an HTML template to canvas,
 * then embedding it as an image in the PDF. This ensures Hebrew
 * text renders correctly using the browser's font engine.
 */
export function exportAnalysisPdf(
  text: string,
  analysis: AnalyzeResponse,
): void {
  // Create a temporary hidden div with the report content
  const container = document.createElement('div');
  container.style.cssText = `
    position: fixed; top: -9999px; left: -9999px;
    width: 700px; padding: 40px;
    background: white; color: #102a43;
    font-family: 'Heebo', system-ui, sans-serif;
    font-size: 14px; line-height: 1.7;
    direction: rtl;
  `;

  const scoreKeys = ['difficulty', 'style', 'fluency', 'cohesion', 'complexity'] as const;

  const scoresHtml = scoreKeys
    .map((key) => {
      const val = analysis.scores[key];
      const pct = (val * 100).toFixed(0);
      const barWidth = Math.max(val * 100, 2);
      return `
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
          <span style="width:80px;font-weight:600;color:#486581;">${SCORE_LABELS[key]}</span>
          <div style="flex:1;height:8px;background:#e8e0d8;border-radius:4px;overflow:hidden;">
            <div style="width:${barWidth}%;height:100%;background:linear-gradient(90deg,#199473,#27ab83);border-radius:4px;"></div>
          </div>
          <span style="width:40px;text-align:left;font-weight:700;color:#102a43;">${pct}%</span>
        </div>`;
    })
    .join('');

  const diagnosesHtml = analysis.diagnoses
    .map(
      (d) => `
      <div style="border-right:3px solid ${d.severity > 0.7 ? '#e12d39' : d.severity > 0.4 ? '#f0b429' : '#27ab83'};padding:10px 14px;margin-bottom:10px;background:#faf8f5;border-radius:8px;">
        <div style="font-weight:700;color:#102a43;margin-bottom:4px;">${d.label_he} <span style="color:#829ab1;font-weight:400;">(${(d.severity * 100).toFixed(0)}%)</span></div>
        <div style="color:#486581;font-size:13px;">${d.explanation_he}</div>
        ${d.tip_he ? `<div style="color:#147d64;font-size:12px;margin-top:4px;">💡 ${d.tip_he}</div>` : ''}
      </div>`,
    )
    .join('');

  const interventionsHtml = analysis.interventions
    .map(
      (iv) => `
      <div style="padding:8px 0;border-bottom:1px solid #e8e0d8;">
        <div style="font-weight:600;color:#102a43;margin-bottom:4px;">${iv.actions_he[0] || iv.type}</div>
        ${iv.actions_he
          .map((a) => `<div style="color:#486581;font-size:13px;padding-right:12px;">• ${a}</div>`)
          .join('')}
      </div>`,
    )
    .join('');

  // Truncate text for the PDF
  const displayText = text.length > 1500 ? text.slice(0, 1500) + '…' : text;

  container.innerHTML = `
    <div style="text-align:center;margin-bottom:24px;">
      <h1 style="font-size:22px;color:#102a43;margin:0 0 4px 0;font-family:'Frank Ruhl Libre','Heebo',serif;">מאמן כתיבה בעברית — דוח ניתוח</h1>
      <div style="color:#829ab1;font-size:12px;">${new Date().toLocaleDateString('he-IL')}</div>
    </div>
    <hr style="border:none;border-top:1px solid #e8e0d8;margin:16px 0;">

    <h2 style="font-size:16px;color:#102a43;margin:0 0 12px 0;">ציונים</h2>
    ${scoresHtml}
    <hr style="border:none;border-top:1px solid #e8e0d8;margin:16px 0;">

    ${analysis.diagnoses.length > 0 ? `
      <h2 style="font-size:16px;color:#102a43;margin:0 0 12px 0;">אבחנות</h2>
      ${diagnosesHtml}
      <hr style="border:none;border-top:1px solid #e8e0d8;margin:16px 0;">
    ` : ''}

    ${analysis.interventions.length > 0 ? `
      <h2 style="font-size:16px;color:#102a43;margin:0 0 12px 0;">המלצות</h2>
      ${interventionsHtml}
      <hr style="border:none;border-top:1px solid #e8e0d8;margin:16px 0;">
    ` : ''}

    <h2 style="font-size:16px;color:#102a43;margin:0 0 12px 0;">הטקסט המנותח</h2>
    <div style="color:#486581;font-size:13px;line-height:1.8;white-space:pre-wrap;">${displayText}</div>
  `;

  document.body.appendChild(container);

  html2canvas(container, {
    scale: 2,
    useCORS: true,
    backgroundColor: '#ffffff',
  }).then((canvas) => {
    document.body.removeChild(container);

    const imgData = canvas.toDataURL('image/png');
    const imgWidth = 190; // A4 width minus margins
    const imgHeight = (canvas.height * imgWidth) / canvas.width;

    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4',
    });

    const pageHeight = pdf.internal.pageSize.getHeight() - 20; // margin
    let position = 10;
    let remainingHeight = imgHeight;

    // First page
    pdf.addImage(imgData, 'PNG', 10, position, imgWidth, imgHeight);

    // Add more pages if content overflows
    while (remainingHeight > pageHeight) {
      remainingHeight -= pageHeight;
      position -= pageHeight;
      pdf.addPage();
      pdf.addImage(imgData, 'PNG', 10, position, imgWidth, imgHeight);
    }

    pdf.save('hebrew-writing-analysis.pdf');
  }).catch(() => {
    document.body.removeChild(container);
    alert('שגיאה ביצירת ה-PDF');
  });
}
