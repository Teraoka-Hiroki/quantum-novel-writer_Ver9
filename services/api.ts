import { AppState, OptimizationResult } from '../types';

const API_BASE = '/api';

export const generateCandidates = async (data: Partial<AppState>) => {
  const res = await fetch(`${API_BASE}/generate_candidates`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return res.json();
};

export const runOptimization = async (data: Partial<AppState>) => {
  const res = await fetch(`${API_BASE}/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return res.json();
};

export const runBBOIteration = async (data: Partial<AppState>) => {
  const res = await fetch(`${API_BASE}/bbo_step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return res.json();
};

export const resetBBO = async () => {
  const res = await fetch(`${API_BASE}/bbo_reset`, {
    method: 'POST',
  });
  return res.json();
};

export const generateDraft = async () => {
  const res = await fetch(`${API_BASE}/generate_draft`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
  return res.json();
};

export const saveDraftEdit = async (article: string, instruction: string) => {
  const res = await fetch(`${API_BASE}/save_draft_edit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ article, instruction }),
  });
  return res.json();
};

export const generateFinal = async () => {
  const res = await fetch(`${API_BASE}/generate_final`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
  return res.json();
};

export const uploadSettings = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/settings/upload`, {
    method: 'POST',
    body: formData,
  });
  return res.json();
};