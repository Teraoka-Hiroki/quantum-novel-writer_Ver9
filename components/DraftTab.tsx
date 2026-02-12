import React from 'react';
import { AppState } from '../types';
import { Pencil, FileText, MessageSquare } from 'lucide-react';

interface Props {
  state: AppState;
  generateDraft: () => void;
  updateState: (updates: Partial<AppState>) => void;
  isLoading: boolean;
}

export const DraftTab: React.FC<Props> = ({ state, generateDraft, updateState, isLoading }) => {
  return (
    <div className="animate-fadeIn">
      <div className="flex justify-between items-center mb-6">
          <h5 className="text-xl font-bold text-gray-200">ドラフト生成</h5>
          <button 
            onClick={generateDraft}
            disabled={isLoading}
            className="bg-cyan-600 hover:bg-cyan-500 text-white px-6 py-2 rounded-full font-bold shadow-lg transition disabled:opacity-50 flex items-center"
          >
            {isLoading ? <span className="animate-spin mr-2">⟳</span> : <Pencil className="w-4 h-4 mr-2" />}
            ドラフトを生成
          </button>
      </div>

      <div className="space-y-6">
          <div>
              <label className="text-sm font-bold text-gray-400 mb-2 block flex items-center">
                  <FileText className="w-4 h-4 mr-2" /> 1. あらすじ / プロット
              </label>
              <textarea 
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg p-4 text-gray-300 focus:border-cyan-500 focus:outline-none h-32"
                  readOnly
                  value={state.draft_summary}
              />
          </div>
          <div>
              <label className="text-sm font-bold text-gray-400 mb-2 block flex items-center">
                  <FileText className="w-4 h-4 mr-2" /> 2. 生成されたシーン (編集可能)
              </label>
              <textarea 
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg p-4 text-gray-200 focus:border-cyan-500 focus:outline-none h-96 font-serif leading-loose"
                  value={state.draft_article}
                  onChange={(e) => updateState({ draft_article: e.target.value })}
              />
          </div>
          <div>
              <label className="text-sm font-bold text-gray-400 mb-2 block flex items-center">
                  <MessageSquare className="w-4 h-4 mr-2" /> 3. 最終仕上げへの指示
              </label>
              <textarea 
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg p-4 text-gray-300 focus:border-cyan-500 focus:outline-none h-24"
                  placeholder="例: 心理描写を深くし、ハードボイルドな文体で..."
                  value={state.additional_instruction}
                  onChange={(e) => updateState({ additional_instruction: e.target.value })}
              />
          </div>
      </div>
    </div>
  );
};