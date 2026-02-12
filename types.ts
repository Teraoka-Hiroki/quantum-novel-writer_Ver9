export interface Params {
  p_desc_style: number;
  p_perspective: number;
  p_sensory: number;
  p_thought: number;
  p_tension: number;
  p_reality: number;
  p_char_count: number;
  p_char_mental: number;
  p_char_belief: number;
  p_char_trauma: number;
  p_char_voice: number;
  length: number;
}

export interface CandidateAttributes {
  [key: string]: number;
}

export interface Candidate {
  id: number;
  text: string;
  type: 'Scene Craft' | 'Character Dynamics';
  relevance: number;
  attributes: CandidateAttributes;
  selected: boolean;
  user_rating: number; // 0 for unrated, 1-5 for rated
}

export interface OptimizationHistoryItem {
  time: number;
  value: number;
}

export interface OptimizationResult {
  candidates: Candidate[];
  history_count?: number;
  plot_data?: OptimizationHistoryItem[];
  scales?: {
      pref?: number;
      diff: number;
      constraint: number;
  };
}

export interface AppState {
  gemini_key: string;
  amplify_token: string;
  topic_main: string;
  topic_sub1: string;
  topic_sub2: string;
  params: Params;
  candidates: Candidate[];
  bbo_history_count: number;
  draft_summary: string;
  draft_article: string;
  additional_instruction: string;
  final_text: string;
  optimization_plot_data: OptimizationHistoryItem[];
  optimization_scales?: {
      pref?: number;
      diff: number;
      constraint: number;
  };
}

export const DEFAULT_PARAMS: Params = {
  p_desc_style: 0.5,
  p_perspective: 0.5,
  p_sensory: 0.5,
  p_thought: 0.5,
  p_tension: 0.5,
  p_reality: 0.5,
  p_char_count: 0.2,
  p_char_mental: 0.5,
  p_char_belief: 0.5,
  p_char_trauma: 0.0,
  p_char_voice: 0.5,
  length: 500,
};