import React from 'react';
import { AppState, Candidate } from '../types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { RotateCcw, Send, Cpu, Layers, Target, AlignJustify } from 'lucide-react';

interface Props {
  state: AppState;
  updateState: (updates: Partial<AppState>) => void;
  updateCandidateRating: (id: number, rating: number) => void;
  runBBO: () => void;
  runLegacy: () => void; // 【修正】App.tsxに合わせて名前を変更
  resetBBO: () => void;
  isLoading: boolean;
  plotData: any[]; // 【追加】グラフデータを受け取る
}

export const OptimizationTab: React.FC<Props> = ({ 
  state, 
  updateState,
  updateCandidateRating, 
  runBBO, 
  runLegacy, // 【修正】名前を変更
  resetBBO, 
  isLoading, 
  plotData // 【追加】
}) => {
  
  // 現在選択されているブロックの統計情報を計算
  const selectedCandidates = state.candidates.filter(c => c.selected);
  const currentLength = selectedCandidates.reduce((sum, c) => sum + c.text.length, 0);
  const targetLength = state.params.length;
  const diffLength = currentLength - targetLength;
  
  const renderCandidateGroup = (type: 'Scene Craft' | 'Character Dynamics') => {
    const group = state.candidates.filter(c => c.type === type);
    if (group.length === 0) return null;
    
    const displayType = type === 'Scene Craft' ? 'シーン描写' : 'キャラクター造形';

    return (
      <div className="mb-6" key={type}>
        <div className="bg-gradient-to-r from-slate-700 to-slate-800 text-white px-4 py-2 rounded-lg mb-3 flex items-center font-bold shadow-md">
            <Layers className="w-4 h-4 mr-2" /> {displayType}
        </div>
        <div className="space-y-3">
            {group.map(item => (
                <div 
                    key={item.id} 
                    className={`relative p-4 rounded-xl border transition-all duration-200 ${
                        item.selected 
                        ? 'bg-slate-800 border-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.3)] border-l-4' 
                        : 'bg-slate-800 border-transparent border-l-4 border-l-transparent hover:translate-y-[-2px] hover:shadow-lg'
                    }`}
                >
                    <div className="flex justify-between items-start mb-2">
                        <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider text-white ${
                            type === 'Scene Craft' ? 'bg-emerald-700' : 'bg-amber-800'
                        }`}>
                            {displayType}
                        </span>
                        <div className="flex items-center bg-slate-900 rounded-full px-3 py-1 gap-2">
                            <span className="text-xs text-gray-500">評価:</span>
                            {[1, 2, 3, 4, 5].map(r => (
                                <label key={r} className="cursor-pointer flex items-center text-xs">
                                    <input 
                                        type="radio" 
                                        name={`rating-${item.id}`} 
                                        className="mr-1 accent-blue-500"
                                        checked={item.user_rating === r}
                                        onChange={() => updateCandidateRating(item.id, r)}
                                    />
                                    <span className={item.user_rating === r ? 'text-blue-400 font-bold' : 'text-gray-500'}>{r}</span>
                                </label>
                            ))}
                        </div>
                    </div>
                    
                    <div className="flex flex-wrap gap-2 mb-2 pb-2 border-b border-slate-700 text-xs text-gray-400">
                        <span className="font-bold text-blue-400 flex items-center">
                            <Target className="w-3 h-3 mr-1" /> 適合度: {item.relevance.toFixed(2)}
                        </span>
                        <span className="bg-slate-700/50 px-2 py-0.5 rounded-full flex items-center">
                            <AlignJustify className="w-3 h-3 mr-1" /> {item.text.length}文字
                        </span>
                        {Object.entries(item.attributes).map(([k, v]) => (
                            <span key={k} className="bg-slate-700/50 px-2 py-0.5 rounded-full">
                                <span className="mr-1 font-semibold text-gray-300">{k.replace('p_', '').replace('char_', '')}</span>
                                {(v as number).toFixed(2)}
                            </span>
                        ))}
                    </div>

                    <p className="text-sm text-gray-200 leading-relaxed">{item.text}</p>
                </div>
            ))}
        </div>
      </div>
    );
  };

  return (
    <div className="animate-fadeIn">
        {/* BBO Control Panel */}
        <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-6 border border-slate-600 mb-6 shadow-xl">
            <div className="flex justify-between items-center mb-4">
                <h5 className="text-xl font-bold text-blue-400 flex items-center">
                    <Cpu className="w-6 h-6 mr-2" /> Human-in-the-Loop 最適化
                </h5>
                <button 
                    onClick={resetBBO}
                    className="text-red-400 hover:text-red-300 text-xs flex items-center border border-red-900/50 bg-red-900/10 px-3 py-1 rounded hover:bg-red-900/30 transition"
                >
                    <RotateCcw className="w-3 h-3 mr-1" /> 学習リセット
                </button>
            </div>
            <p className="text-sm text-gray-300 mb-4 leading-relaxed">
                ブラックボックス最適化（BBO）を使用して、生成されたブロックの組み合わせをあなたの好みに基づいて最適化します。<br/>
                生成されたブロックを評価してください（1: 拒否 - 5: 採用）。<br/>
                「学習して最適化」をクリックすると、あなたの好みに基づいて量子アニーリングマシンが最適な組み合わせを探索します。
            </p>
            <div className="flex justify-between items-end">
                <div className="text-xs text-gray-400">
                    <span className="bg-slate-900 px-2 py-1 rounded text-gray-300">
                        学習データ数: <span className="text-blue-400 font-bold">{state.bbo_history_count}</span>
                    </span>
                </div>
                <button 
                    onClick={runBBO}
                    disabled={isLoading}
                    className="bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-2 px-6 rounded-lg shadow-lg flex items-center transition disabled:opacity-50"
                >
                    {isLoading ? <span className="animate-spin mr-2">⟳</span> : <Send className="w-4 h-4 mr-2" />}
                    学習して最適化
                </button>
            </div>
            
            {state.optimization_scales && (
                <div className="mt-4 pt-3 border-t border-slate-600/50 flex flex-wrap gap-4 text-xs text-gray-300 items-center">
                    <span className="font-semibold">最適化ステータス:</span>
                    
                    <div className={`px-2 py-0.5 rounded flex items-center border ${Math.abs(diffLength) < 10 ? 'bg-emerald-900/30 border-emerald-500/30 text-emerald-300' : 'bg-amber-900/30 border-amber-500/30 text-amber-300'}`}>
                        <AlignJustify className="w-3 h-3 mr-1" />
                        合計文字数: <strong className="mx-1 text-sm">{currentLength}</strong> / {targetLength} 
                        (差: {diffLength > 0 ? `+${diffLength}` : diffLength})
                    </div>

                    <span className="bg-blue-900/30 px-2 py-0.5 rounded text-blue-300 border border-blue-500/20">
                        好み(Pref): {state.optimization_scales.pref !== undefined ? state.optimization_scales.pref.toFixed(2) : '-'}
                    </span>
                    <span className="bg-purple-900/30 px-2 py-0.5 rounded text-purple-300 border border-purple-500/20">
                        属性差分(Diff): {state.optimization_scales.diff.toFixed(2)}
                    </span>
                    <span className="bg-red-900/30 px-2 py-0.5 rounded text-red-300 border border-red-500/20">
                        制約(Constraint): {state.optimization_scales.constraint.toFixed(4)}
                    </span>
                </div>
            )}
        </div>

        {/* List Header */}
        <div className="flex justify-between items-center mb-4 border-b border-slate-700 pb-2">
            <div>
                <h5 className="font-bold text-gray-200">候補ブロック</h5>
                <p className="text-xs text-gray-500">青くハイライトされたカードが現在の最適な選択です。</p>
            </div>
            <button 
                onClick={runLegacy} // 【修正】正しい関数名を使用
                disabled={isLoading}
                className="text-xs border border-slate-600 text-gray-400 hover:text-white px-3 py-1 rounded-full transition flex items-center"
            >
                <Cpu className="w-3 h-3 mr-1" /> パラメータのみで最適化
            </button>
        </div>

        {/* Candidates */}
        {state.candidates.length === 0 ? (
            <div className="text-center py-10 text-gray-500 bg-slate-800/50 rounded-lg">
                候補がありません。タブ1で生成してください。
            </div>
        ) : (
            <div>
                {renderCandidateGroup('Scene Craft')}
                {renderCandidateGroup('Character Dynamics')}
            </div>
        )}

        {/* Graph Section */}
        <div className="mt-8 pt-6 border-t border-slate-700">
            <h6 className="text-blue-300 font-semibold mb-4 flex items-center">
                <Target className="w-4 h-4 mr-2" /> BBO 最適化グラフ
            </h6>
            <div className="bg-slate-900 rounded-lg p-4 h-[300px] flex items-center justify-center border border-slate-800">
                {/* 【修正】state.optimization_plot_data ではなく prop の plotData を使用 */}
                {plotData && plotData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={plotData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis 
                                dataKey="time" 
                                type="number" 
                                domain={['dataMin', 'dataMax']}
                                label={{ value: '時間 (s)', position: 'insideBottomRight', offset: -5, fill: '#94a3b8' }} 
                                stroke="#94a3b8"
                                tickFormatter={(tick) => tick.toFixed(2)}
                            />
                            <YAxis 
                                stroke="#94a3b8" 
                                label={{ value: '目的関数値', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                                domain={['auto', 'auto']}
                            />
                            <Tooltip 
                                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', color: '#f1f5f9' }}
                                itemStyle={{ color: '#60a5fa' }}
                            />
                            <Legend />
                            <Line type="monotone" dataKey="value" stroke="#60a5fa" strokeWidth={2} dot={{ r: 3, fill: '#60a5fa' }} animationDuration={500} name="目的関数値" />
                        </LineChart>
                    </ResponsiveContainer>
                ) : (
                    <p className="text-gray-400 text-sm font-medium">
                        最適化を実行すると、経過時間と目的関数値のグラフがここに表示されます。
                    </p>
                )}
            </div>
        </div>
    </div>
  );
};