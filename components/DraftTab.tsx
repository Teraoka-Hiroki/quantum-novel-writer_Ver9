import React from 'react';
import { AppState } from '../types';
import { Pencil, Save, Play } from 'lucide-react';

interface Props {
  state: AppState;
  generateDraft: (key?: string) => void;
  updateState: (updates: Partial<AppState>) => void;
  isLoading: boolean;
}

export const DraftTab: React.FC<Props> = ({ state, generateDraft, updateState, isLoading }) => {
  
  const handleSaveDraft = async () => {
      // In a real app, this might just update state or call an API to save intermediate work
      // Here we assume state is already updated via textarea onChange, 
      // but we can trigger an explicit "save" API if needed.
      // For now, we rely on the fact that generating final will save it.
      // But let's add a visual feedback or explicit save call if desired.
      // The current flow saves on "Generate Final", but we can add an explicit save.
      try {
          await fetch('/api/save_draft_edit', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({
                  article: state.draft_article,
                  instruction: state.additional_instruction
              })
          });
          alert("下書きを一時保存しました。");
      } catch(e) {
          alert("保存に失敗しました。");
      }
  };

  return (
    <div className="animate-fadeIn">
      <div className="flex justify-between items-center mb-6">
          <h5 className="text-xl font-bold text-gray-200">ドラフト生成</h5>
          <button 
            onClick={() => generateDraft(state.gemini_key)}
            disabled={isLoading}
            className="bg-cyan-600 hover:bg-cyan-500 text-white px-6 py-2 rounded-full font-bold shadow-lg transition disabled:opacity-50 flex items-center"
          >
            {isLoading ? <span className="animate-spin mr-2">⟳</span> : <Pencil className="w-4 h-4 mr-2" />}
            ドラフトを生成
          </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[600px]">
        {/* Left: Summary & Instruction */}
        <div className="flex flex-col gap-4 h-full">
            <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 flex-1 flex flex-col">
                <h6 className="text-gray-400 font-bold mb-2 flex-shrink-0">あらすじ (自動生成)</h6>
                <textarea 
                    className="w-full bg-slate-900 border border-slate-700 rounded p-3 text-gray-300 text-sm flex-grow resize-none focus:outline-none focus:border-cyan-500"
                    value={state.draft_summary}
                    readOnly
                />
            </div>
            <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 flex-1 flex flex-col">
                <h6 className="text-gray-400 font-bold mb-2 flex-shrink-0">追加指示 (推敲・補正用)</h6>
                <textarea 
                    className="w-full bg-slate-900 border border-slate-700 rounded p-3 text-gray-300 text-sm flex-grow resize-none focus:outline-none focus:border-cyan-500"
                    placeholder="例: もう少し心理描写を増やして、結末を衝撃的にしてください。"
                    value={state.additional_instruction}
                    onChange={(e) => updateState({ additional_instruction: e.target.value })}
                />
            </div>
        </div>

        {/* Right: Draft Editor */}
        <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 h-full flex flex-col">
             <div className="flex justify-between items-center mb-2 flex-shrink-0">
                <h6 className="text-gray-400 font-bold">ドラフト本文 (編集可能)</h6>
                <button 
                    onClick={handleSaveDraft}
                    className="text-xs flex items-center gap-1 bg-slate-700 hover:bg-slate-600 text-gray-300 px-3 py-1 rounded transition"
                >
                    <Save className="w-3 h-3" /> 保存
                </button>
             </div>
             <textarea 
                className="w-full bg-slate-900 border border-slate-700 rounded p-4 text-gray-200 leading-relaxed flex-grow resize-none focus:outline-none focus:border-cyan-500 font-serif"
                value={state.draft_article}
                onChange={(e) => updateState({ draft_article: e.target.value })}
             />
        </div>
      </div>
    </div>
  );
};