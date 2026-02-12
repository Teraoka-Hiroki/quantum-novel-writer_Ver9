import React from 'react';
import { AppState } from '../types';
import { Download, Wand2 } from 'lucide-react';

interface Props {
  state: AppState;
  generateFinal: () => void;
  isLoading: boolean;
  updateState: (updates: Partial<AppState>) => void;
}

export const FinalTab: React.FC<Props> = ({ state, generateFinal, isLoading, updateState }) => {
  
  const handleDownload = () => {
    const element = document.createElement("a");
    const file = new Blob([state.final_text], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = "novel_scene.txt";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div className="animate-fadeIn">
       <div className="flex justify-between items-center mb-6">
          <h5 className="text-xl font-bold text-gray-200">最終出力</h5>
          <div className="flex gap-2">
            <button 
                onClick={handleDownload}
                className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-full font-bold transition flex items-center"
            >
                <Download className="w-4 h-4 mr-2" /> 保存 (.txt)
            </button>
            <button 
                onClick={generateFinal}
                disabled={isLoading}
                className="bg-amber-500 hover:bg-amber-400 text-slate-900 px-6 py-2 rounded-full font-bold shadow-lg transition disabled:opacity-50 flex items-center"
            >
                {isLoading ? <span className="animate-spin mr-2">⟳</span> : <Wand2 className="w-4 h-4 mr-2" />}
                最終仕上げを実行
            </button>
          </div>
      </div>
      
      <textarea 
          className="w-full bg-[#1e2a35] border border-slate-700 rounded-lg p-6 text-blue-100 focus:border-amber-500 focus:outline-none h-[600px] font-mono leading-relaxed"
          value={state.final_text}
          onChange={(e) => updateState({ final_text: e.target.value })}
          placeholder="ここに最終的な文章が表示されます。編集可能です。"
      />
    </div>
  );
};