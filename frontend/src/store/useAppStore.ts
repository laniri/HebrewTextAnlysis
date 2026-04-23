import { create } from 'zustand';
import type {
  AnalyzeResponse,
  SentenceAnnotation,
  AdminConfig,
} from '../types';
import {
  analyzeText as apiAnalyze,
  getAdminConfig as apiGetConfig,
  updateAdminConfig as apiUpdateConfig,
} from '../api/client';

interface RewriteModalState {
  isOpen: boolean;
  targetSentences: SentenceAnnotation[];
  diagnosisType: string;
  suggestion: string | null;
  revisedText: string;
  deltaScores: Record<string, number> | null;
}

const initialRewriteModal: RewriteModalState = {
  isOpen: false,
  targetSentences: [],
  diagnosisType: '',
  suggestion: null,
  revisedText: '',
  deltaScores: null,
};

export interface AppState {
  // Editor
  text: string;
  setText: (text: string) => void;

  // Analysis
  analysisResult: AnalyzeResponse | null;
  isAnalyzing: boolean;
  analyzeText: (text: string) => Promise<void>;

  // Previous analysis (for progress feedback)
  previousAnalysis: AnalyzeResponse | null;

  // Rewrite modal
  rewriteModal: RewriteModalState;
  openRewriteModal: (
    sentences: SentenceAnnotation[],
    diagnosisType: string,
    suggestion?: string,
  ) => void;
  closeRewriteModal: () => void;
  updateRevisedText: (text: string) => void;
  setDeltaScores: (deltas: Record<string, number> | null) => void;
  applyRewrite: () => void;

  // Admin
  adminConfig: AdminConfig | null;
  fetchAdminConfig: (password: string) => Promise<void>;
  updateAdminConfig: (
    password: string,
    config: Partial<AdminConfig>,
  ) => Promise<void>;
}

export const useAppStore = create<AppState>((set, get) => ({
  // ── Editor ──────────────────────────────────────────────
  text: '',
  setText: (text: string) => set({ text }),

  // ── Analysis ────────────────────────────────────────────
  analysisResult: null,
  isAnalyzing: false,
  previousAnalysis: null,

  analyzeText: async (text: string) => {
    set({ isAnalyzing: true });
    try {
      const result = await apiAnalyze(text);
      set((state) => ({
        analysisResult: result,
        previousAnalysis: state.analysisResult,
        isAnalyzing: false,
      }));
    } catch {
      set({ isAnalyzing: false });
    }
  },

  // ── Rewrite modal ───────────────────────────────────────
  rewriteModal: { ...initialRewriteModal },

  openRewriteModal: (
    sentences: SentenceAnnotation[],
    diagnosisType: string,
    suggestion?: string,
  ) => {
    const targetText = sentences.map((s) => s.text).join(' ');
    set({
      rewriteModal: {
        isOpen: true,
        targetSentences: sentences,
        diagnosisType,
        suggestion: suggestion ?? null,
        revisedText: suggestion ?? '',
        deltaScores: null,
      },
    });
  },

  closeRewriteModal: () => {
    set({ rewriteModal: { ...initialRewriteModal } });
  },

  updateRevisedText: (text: string) => {
    set((state) => ({
      rewriteModal: { ...state.rewriteModal, revisedText: text },
    }));
  },

  setDeltaScores: (deltas: Record<string, number> | null) => {
    set((state) => ({
      rewriteModal: { ...state.rewriteModal, deltaScores: deltas },
    }));
  },

  applyRewrite: () => {
    const { rewriteModal, text } = get();
    if (!rewriteModal.isOpen || rewriteModal.targetSentences.length === 0) {
      return;
    }

    // Sort target sentences by char_start so we replace from end to start
    // to preserve earlier offsets.
    const sorted = [...rewriteModal.targetSentences].sort(
      (a, b) => a.char_start - b.char_start,
    );

    const rangeStart = sorted[0].char_start;
    const rangeEnd = sorted[sorted.length - 1].char_end;

    const newText =
      text.slice(0, rangeStart) +
      rewriteModal.revisedText +
      text.slice(rangeEnd);

    set({
      text: newText,
      rewriteModal: { ...initialRewriteModal },
    });

    // Trigger re-analysis on the updated text
    get().analyzeText(newText);
  },

  // ── Admin ───────────────────────────────────────────────
  adminConfig: null,

  fetchAdminConfig: async (password: string) => {
    const config = await apiGetConfig(password);
    set({ adminConfig: config });
  },

  updateAdminConfig: async (
    password: string,
    config: Partial<AdminConfig>,
  ) => {
    const updated = await apiUpdateConfig(password, config);
    set({ adminConfig: updated });
  },
}));
