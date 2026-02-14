import { AppState, OptimizationResult } from '../types';

const API_BASE = '/api';

export const generateCandidates = async (
  gemini_key: string,
  topic_main: string,
  topic_sub1: string,
  topic_sub2: string,
  params: AppState['params'],
  target_type?: string, // 追加
  append: boolean = false // 追加
) => {
  const res = await fetch(`${API_BASE}/generate_candidates`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ gemini_key, topic_main, topic_sub1, topic_sub2, params, target_type, append }),
  });
  return res.json();
};

export const updateCandidateRating = async (id: number, rating: number) => {
  const res = await fetch(`${API_BASE}/update_rating`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id, rating }),
  });
  return res.json();
};

export const runOptimization = async (amplify_token: string, params: AppState['params']) => {
  const res = await fetch(`${API_BASE}/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ amplify_token, params }),
  });
  return res.json();
};

export const runBBOIteration = async (amplify_token: string, params: AppState['params']) => {
  const res = await fetch(`${API_BASE}/bbo_step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ amplify_token, params }),
  });
  return res.json();
};

export const resetBBO = async () => {
  const res = await fetch(`${API_BASE}/bbo_reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
  return res.json();
};

export const generateDraft = async (data?: { gemini_key?: string }) => {
  const res = await fetch(`${API_BASE}/generate_draft`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data || {}),
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

export const generateFinal = async (data?: { gemini_key?: string }) => {
  const res = await fetch(`${API_BASE}/generate_final`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data || {}),
  });
  return res.json();
};

export const uploadSettings = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/settings/upload`, {
        method: 'POST',
        body: formData
    });
    return res.json();
};