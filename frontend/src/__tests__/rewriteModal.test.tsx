import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useAppStore } from '../store/useAppStore';
import RewriteModal from '../components/RewriteModal';
import type { SentenceAnnotation } from '../types';

// Mock diff-match-patch
vi.mock('diff-match-patch', () => ({
  diff_match_patch: class {
    diff_main(a: string, b: string) {
      if (a === b) return [[0, a]];
      return [[-1, a], [1, b]];
    }
    diff_cleanupSemantic() {}
  },
}));

// Mock the API client
vi.mock('../api/client', () => ({
  analyzeText: vi.fn(),
  reviseText: vi.fn().mockResolvedValue({ deltas: {} }),
  getAdminConfig: vi.fn(),
  updateAdminConfig: vi.fn(),
}));

const mockSentences: SentenceAnnotation[] = [
  {
    index: 0,
    text: 'זהו משפט לדוגמה',
    char_start: 0,
    char_end: 16,
    complexity: 0.5,
    highlight: 'yellow',
  },
];

describe('Task 13.2: RewriteModal', () => {
  beforeEach(() => {
    // Reset store to initial state
    useAppStore.setState({
      text: 'זהו משפט לדוגמה',
      rewriteModal: {
        isOpen: false,
        targetSentences: [],
        diagnosisType: '',
        suggestion: null,
        revisedText: '',
        deltaScores: null,
      },
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should render the modal when isOpen is true', () => {
    useAppStore.setState({
      rewriteModal: {
        isOpen: true,
        targetSentences: mockSentences,
        diagnosisType: 'low_cohesion',
        suggestion: null,
        revisedText: 'זהו משפט לדוגמה',
        deltaScores: null,
      },
    });

    render(<RewriteModal />);

    // Modal should be visible with the dialog role
    const dialog = screen.getByRole('dialog');
    expect(dialog).toBeDefined();

    // Should show the heading
    expect(screen.getByText('שכתוב מונחה')).toBeDefined();

    // Should show the original text label
    expect(screen.getByText('טקסט מקורי')).toBeDefined();
  });

  it('should NOT render the modal when isOpen is false', () => {
    useAppStore.setState({
      rewriteModal: {
        isOpen: false,
        targetSentences: [],
        diagnosisType: '',
        suggestion: null,
        revisedText: '',
        deltaScores: null,
      },
    });

    render(<RewriteModal />);

    // Modal should not be in the DOM
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('should call applyRewrite when Apply button is clicked', async () => {
    const applyRewriteSpy = vi.fn();

    useAppStore.setState({
      text: 'זהו משפט לדוגמה',
      rewriteModal: {
        isOpen: true,
        targetSentences: mockSentences,
        diagnosisType: 'low_cohesion',
        suggestion: null,
        revisedText: 'זהו משפט משופר',
        deltaScores: null,
      },
      applyRewrite: applyRewriteSpy,
    });

    const user = userEvent.setup();
    render(<RewriteModal />);

    const applyButton = screen.getByText('החל');
    await user.click(applyButton);

    expect(applyRewriteSpy).toHaveBeenCalledTimes(1);
  });

  it('should call closeRewriteModal when Cancel button is clicked', async () => {
    const closeRewriteModalSpy = vi.fn();

    useAppStore.setState({
      rewriteModal: {
        isOpen: true,
        targetSentences: mockSentences,
        diagnosisType: 'low_cohesion',
        suggestion: null,
        revisedText: 'זהו משפט לדוגמה',
        deltaScores: null,
      },
      closeRewriteModal: closeRewriteModalSpy,
    });

    const user = userEvent.setup();
    render(<RewriteModal />);

    const cancelButton = screen.getByText('בטל');
    await user.click(cancelButton);

    expect(closeRewriteModalSpy).toHaveBeenCalledTimes(1);
  });
});
