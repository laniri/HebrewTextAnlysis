export interface ScoresResponse {
  difficulty: number;
  style: number;
  fluency: number;
  cohesion: number;
  complexity: number;
}

export interface SentenceAnnotation {
  index: number;
  text: string;
  char_start: number;
  char_end: number;
  complexity: number;
  highlight: "red" | "yellow" | "none";
}

export interface CohesionGap {
  pair: [number, number];
  severity: number;
  char_start: number;
  char_end: number;
}

export interface LocalizedDiagnosis {
  type: string;
  severity: number;
  label_he: string;
  explanation_he: string;
  actions_he: string[];
  tip_he: string;
}

export interface LocalizedIntervention {
  type: string;
  priority: number;
  target_diagnosis: string;
  actions_he: string[];
  exercises_he: string[];
}

export interface AnalyzeResponse {
  scores: ScoresResponse;
  diagnoses: LocalizedDiagnosis[];
  interventions: LocalizedIntervention[];
  sentences: SentenceAnnotation[];
  cohesion_gaps: CohesionGap[];
}

export interface ReviseResponse {
  original_scores: ScoresResponse;
  revised_scores: ScoresResponse;
  deltas: Record<string, number>;
  resolved_diagnoses: string[];
  new_diagnoses: string[];
}

export interface RewriteResponse {
  suggestion: string;
  model_used: string;
}

export interface ExampleSummary {
  id: string;
  label: string;
  category: string;
  preview: string;
}

export interface ExampleFull {
  id: string;
  label: string;
  category: string;
  text: string;
}

export interface AdminConfig {
  bedrock_model_id: string;
  severity_threshold: number;
  max_diagnoses_shown: number;
  max_interventions_shown: number;
}

export interface ModelInfo {
  model_id: string;
  model_name: string;
  provider: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

export interface ExerciseOption {
  text: string;
  is_correct: boolean;
  explanation_he: string;
}

export interface ExerciseResponse {
  original_text: string;
  diagnosis_label_he: string;
  diagnosis_explanation_he: string;
  tip_he: string;
  options: ExerciseOption[];
}
