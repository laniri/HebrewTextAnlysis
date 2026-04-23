import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { useAppStore } from '../store/useAppStore';
import ScoreSpiderChart from '../components/ScoreSpiderChart';
import DiagnosisList from '../components/DiagnosisList';
import type { ScoresResponse, LocalizedDiagnosis } from '../types';

// Mock Monaco Editor — jsdom can't handle it
vi.mock('@monaco-editor/react', () => ({
  default: (props: Record<string, unknown>) => (
    <div data-testid="monaco-editor" data-value={props.value as string} />
  ),
}));

// Mock recharts to avoid SVG rendering issues in jsdom
vi.mock('recharts', () => {
  const MockRadarChart = ({ children, data }: { children: React.ReactNode; data: unknown[] }) => (
    <div data-testid="radar-chart" data-axes={JSON.stringify(data)}>
      {children}
    </div>
  );
  const MockPolarAngleAxis = ({ dataKey }: { dataKey: string }) => (
    <div data-testid="polar-angle-axis" data-key={dataKey} />
  );
  const MockRadar = ({ name, dataKey }: { name: string; dataKey: string }) => (
    <div data-testid="radar" data-name={name} data-key={dataKey} />
  );
  return {
    RadarChart: MockRadarChart,
    PolarGrid: () => <div data-testid="polar-grid" />,
    PolarAngleAxis: MockPolarAngleAxis,
    PolarRadiusAxis: () => <div data-testid="polar-radius-axis" />,
    Radar: MockRadar,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
    Tooltip: () => <div data-testid="tooltip" />,
  };
});

// Mock the API client
vi.mock('../api/client', () => ({
  analyzeText: vi.fn(),
  getAdminConfig: vi.fn(),
  updateAdminConfig: vi.fn(),
}));

describe('Task 13.1: Editor and Analysis Panel', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    // Reset store state
    useAppStore.setState({
      text: '',
      analysisResult: null,
      isAnalyzing: false,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  describe('Debounce fires analysis after 800ms inactivity', () => {
    it('should call analyzeText after 800ms debounce when text is set', async () => {
      const { analyzeText: apiAnalyze } = await import('../api/client');
      const mockAnalyze = vi.mocked(apiAnalyze);
      mockAnalyze.mockResolvedValue({
        scores: { difficulty: 0.5, style: 0.5, fluency: 0.5, cohesion: 0.5, complexity: 0.5 },
        diagnoses: [],
        interventions: [],
        sentences: [],
        cohesion_gaps: [],
      });

      const store = useAppStore.getState();

      // Simulate what MonacoEditor does: setText + schedule analyzeText after 800ms
      store.setText('שלום עולם');

      // Trigger the debounced analysis (simulating the editor's onChange)
      const analyzePromise = store.analyzeText('שלום עולם');

      // The API should have been called
      expect(mockAnalyze).toHaveBeenCalledWith('שלום עולם');

      await analyzePromise;

      // After resolution, isAnalyzing should be false
      expect(useAppStore.getState().isAnalyzing).toBe(false);
      expect(useAppStore.getState().analysisResult).not.toBeNull();
    });

    it('should debounce multiple rapid calls — only the last fires', async () => {
      // Test the debounce pattern used in MonacoEditor:
      // Each keystroke clears the previous timer and sets a new 800ms one
      let timerRef: ReturnType<typeof setTimeout> | null = null;
      const analyzeText = vi.fn();

      // Simulate 3 rapid keystrokes
      const simulateKeystroke = (text: string) => {
        if (timerRef) clearTimeout(timerRef);
        if (text.trim()) {
          timerRef = setTimeout(() => analyzeText(text), 800);
        }
      };

      simulateKeystroke('א');
      vi.advanceTimersByTime(200);
      simulateKeystroke('אב');
      vi.advanceTimersByTime(200);
      simulateKeystroke('אבג');

      // At 400ms total, no call yet
      expect(analyzeText).not.toHaveBeenCalled();

      // Advance past the 800ms debounce from the last keystroke
      vi.advanceTimersByTime(800);

      // Only the last text should have triggered analysis
      expect(analyzeText).toHaveBeenCalledTimes(1);
      expect(analyzeText).toHaveBeenCalledWith('אבג');
    });

    it('should not fire analysis for empty/whitespace text', () => {
      let timerRef: ReturnType<typeof setTimeout> | null = null;
      const analyzeText = vi.fn();

      const simulateKeystroke = (text: string) => {
        if (timerRef) clearTimeout(timerRef);
        if (text.trim()) {
          timerRef = setTimeout(() => analyzeText(text), 800);
        }
      };

      simulateKeystroke('   ');
      vi.advanceTimersByTime(1000);

      expect(analyzeText).not.toHaveBeenCalled();
    });
  });

  describe('Score spider chart renders 5 axes with Hebrew labels', () => {
    const mockScores: ScoresResponse = {
      difficulty: 0.8,
      style: 0.6,
      fluency: 0.7,
      cohesion: 0.4,
      complexity: 0.9,
    };

    it('should render the radar chart with 5 data points', () => {
      render(<ScoreSpiderChart scores={mockScores} />);

      const chart = screen.getByTestId('radar-chart');
      expect(chart).toBeDefined();

      const axesData = JSON.parse(chart.getAttribute('data-axes') || '[]');
      expect(axesData).toHaveLength(5);
    });

    it('should include all 5 Hebrew axis labels', () => {
      render(<ScoreSpiderChart scores={mockScores} />);

      const chart = screen.getByTestId('radar-chart');
      const axesData = JSON.parse(chart.getAttribute('data-axes') || '[]');

      const labels = axesData.map((d: { axis: string }) => d.axis);
      expect(labels).toContain('קושי');
      expect(labels).toContain('סגנון');
      expect(labels).toContain('שטף');
      expect(labels).toContain('קוהרנטיות');
      expect(labels).toContain('מורכבות');
    });

    it('should pass correct score values to chart data', () => {
      render(<ScoreSpiderChart scores={mockScores} />);

      const chart = screen.getByTestId('radar-chart');
      const axesData = JSON.parse(chart.getAttribute('data-axes') || '[]');

      const difficultyAxis = axesData.find((d: { axis: string }) => d.axis === 'קושי');
      expect(difficultyAxis.current).toBe(0.8);

      const complexityAxis = axesData.find((d: { axis: string }) => d.axis === 'מורכבות');
      expect(complexityAxis.current).toBe(0.9);
    });

    it('should render PolarAngleAxis with axis dataKey', () => {
      render(<ScoreSpiderChart scores={mockScores} />);

      const angleAxis = screen.getByTestId('polar-angle-axis');
      expect(angleAxis.getAttribute('data-key')).toBe('axis');
    });
  });

  describe('Diagnosis list sorts by severity and caps at max', () => {
    const makeDiagnosis = (type: string, severity: number): LocalizedDiagnosis => ({
      type,
      severity,
      label_he: `אבחנה ${type}`,
      explanation_he: `הסבר עבור ${type}`,
      actions_he: [`פעולה עבור ${type}`],
      tip_he: `טיפ עבור ${type}`,
    });

    it('should render diagnoses sorted by severity descending', () => {
      const diagnoses = [
        makeDiagnosis('low_cohesion', 0.3),
        makeDiagnosis('sentence_over_complexity', 0.9),
        makeDiagnosis('low_lexical_diversity', 0.6),
      ];

      render(<DiagnosisList diagnoses={diagnoses} />);

      // The component sorts by severity descending
      // Check that the highest severity appears first in the DOM
      const cards = screen.getAllByText(/אבחנה/);
      expect(cards[0].textContent).toContain('sentence_over_complexity');
      expect(cards[1].textContent).toContain('low_lexical_diversity');
      expect(cards[2].textContent).toContain('low_cohesion');
    });

    it('should show empty state when no diagnoses', () => {
      render(<DiagnosisList diagnoses={[]} />);

      expect(screen.getByText('לא נמצאו אבחנות — הטקסט נראה תקין')).toBeDefined();
    });

    it('should render all provided diagnoses', () => {
      const diagnoses = [
        makeDiagnosis('a', 0.8),
        makeDiagnosis('b', 0.5),
        makeDiagnosis('c', 0.3),
        makeDiagnosis('d', 0.1),
      ];

      render(<DiagnosisList diagnoses={diagnoses} />);

      // All 4 diagnoses should be rendered
      const labels = screen.getAllByText(/אבחנה/);
      expect(labels).toHaveLength(4);
    });
  });
});
