import React from 'react';
import { AppState } from '../types';
import { Wand2, Download, FileText } from 'lucide-react';

interface Props {
  state: AppState;
  generateFinal: (key?: string) => void;
  isLoading: boolean;
  updateState: (updates: Partial<AppState>) => void;
}

export const FinalTab: React.FC<Props> = ({ state, generateFinal, isLoading, updateState }) => {
  
  const handleDownload = () => {
    // Trigger backend download endpoint
    window.location.href = '/api/download';
  };

  return (
    <div className="animate-fadeIn">
       <div className="flex justify-between items-center mb-6">
          <h5 className="text-xl font-bold text-gray-200">最終出力</h5>
          <div className="flex gap-2">
            <button 
                onClick={handleDownload}
                disabled={!state.final_text}
                className="bg-slate-700 hover:bg-slate-600 text-gray-200 px-4 py-2 rounded-full font-bold shadow transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center text-sm"
            >
                <Download className="w-4 h-4 mr-2" />
                txtで保存
            </button>
            <button 
                onClick={() => generateFinal(state.gemini_key)}
                disabled={isLoading}
                className="bg-amber-500 hover:bg-amber-400 text-slate-900 px-6 py-2 rounded-full font-bold shadow-lg transition disabled:opacity-50 flex items-center"
            >
                {isLoading ? <span className="animate-spin mr-2">⟳</span> : <Wand2 className="w-4 h-4 mr-2" />}
                最終仕上げを実行
            </button>
          </div>
      </div>
      
      <div className="bg-slate-800 rounded-lg p-1 border border-slate-700 shadow-2xl h-[600px] flex flex-col">
          <div className="bg-slate-900/50 p-2 border-b border-slate-700 flex items-center text-gray-400 text-xs">
              <FileText className="w-4 h-4 mr-2" />
              novel_scene.txt
          </div>
          <textarea 
            className="w-full h-full bg-slate-900 text-gray-100 p-6 leading-loose font-serif text-lg resize-none focus:outline-none"
            value={state.final_text}
            // 【修正】readOnlyを削除し、編集可能にする
            onChange={(e) => updateState({ final_text: e.target.value })}
            placeholder="ここに最終的な小説が出力されます..."
          />
      </div>
    </div>
  );
};