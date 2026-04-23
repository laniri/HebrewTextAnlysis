import axios from 'axios';
import type {
  AnalyzeResponse,
  ReviseResponse,
  RewriteResponse,
  ExerciseResponse,
  ExampleSummary,
  ExampleFull,
  AdminConfig,
  ModelInfo,
} from '../types';

const api = axios.create({ baseURL: '/' });

export async function analyzeText(text: string): Promise<AnalyzeResponse> {
  const { data } = await api.post<AnalyzeResponse>('/api/analyze', { text });
  return data;
}

export async function reviseText(
  original_text: string,
  edited_text: string,
): Promise<ReviseResponse> {
  const { data } = await api.post<ReviseResponse>('/api/revise', {
    original_text,
    edited_text,
  });
  return data;
}

export async function rewriteText(
  text: string,
  diagnosis_type: string,
  context: string = '',
): Promise<RewriteResponse> {
  const { data } = await api.post<RewriteResponse>('/api/rewrite', {
    text,
    diagnosis_type,
    context,
  });
  return data;
}

export async function listExamples(): Promise<ExampleSummary[]> {
  const { data } = await api.get<ExampleSummary[]>('/api/examples');
  return data;
}

export async function getExample(id: string): Promise<ExampleFull> {
  const { data } = await api.get<ExampleFull>(`/api/examples/${id}`);
  return data;
}

export async function getAdminConfig(
  password: string,
): Promise<AdminConfig> {
  const { data } = await api.get<AdminConfig>('/admin/config', {
    headers: { 'X-Admin-Password': password },
  });
  return data;
}

export async function updateAdminConfig(
  password: string,
  config: Partial<AdminConfig>,
): Promise<AdminConfig> {
  const { data } = await api.post<AdminConfig>('/admin/config', config, {
    headers: { 'X-Admin-Password': password },
  });
  return data;
}

export async function listModels(
  password: string,
): Promise<ModelInfo[]> {
  const { data } = await api.get<ModelInfo[]>('/admin/models', {
    headers: { 'X-Admin-Password': password },
  });
  return data;
}

export async function generateExercise(
  text: string,
  diagnosis_type: string,
): Promise<ExerciseResponse> {
  const { data } = await api.post<ExerciseResponse>('/api/exercise', {
    text,
    diagnosis_type,
  });
  return data;
}
