import React, { useState, useEffect } from 'react';
import { SettingsTab } from './components/SettingsTab';
import { OptimizationTab } from './components/OptimizationTab';
import { DraftTab } from './components/DraftTab';
import { FinalTab } from './components/FinalTab';
import { AppState, DEFAULT_PARAMS, Params } from './types';
import * as api from './services/api';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingText, setLoadingText] = useState('');
  
  const [state, setState] = useState<AppState>({
    gemini_key: '',
    amplify_token: '',
    topic_main: '',
    topic_sub1: '',
    topic_sub2: '',
    params: DEFAULT_PARAMS,
    candidates: [],
    bbo_history_count: 0,
    draft_summary: '',
    draft_article: '',
    additional_instruction: '',
    final_text: '',
    optimization_plot_data: [],
  });

  const updateState = (updates: Partial<AppState>) => setState(prev => ({ ...prev, ...updates }));
  const updateParams = (updates: Partial<Params>) => setState(prev => ({ ...prev, params: { ...prev.params, ...updates } }));

  const handleGenerateCandidates = async () => {
    if (!state.gemini_key || !state.topic_main) {
      alert("APIキーとメイン設定は必須です。");
      return;
    }
    setIsLoading(true);
    setLoadingText("Geminiで候補を生成中...");
    try {
      const res = await api.generateCandidates({
        gemini_key: state.gemini_key,
        amplify_token: state.amplify_token,
        topic_main: state.topic_main,
        topic_sub1: state.topic_sub1,
        topic_sub2: state.topic_sub2,
        params: state.params
      });
      if (res.status === 'success') {
        updateState({ candidates: res.candidates, bbo_history_count: 0 });
        setActiveTab(1);
      } else {
        alert("エラー: " + res.message);
      }
    } catch (e) {
      alert("通信エラー");
    } finally {
      setIsLoading(false);
    }
  };

  const updateCandidateRating = (id: number, rating: number) => {
    const newCandidates = state.candidates.map(c => c.id === id ? { ...c, user_rating: rating } : c);
    updateState({ candidates: newCandidates });
    
    fetch('/api/update_rating', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ id, rating })
    });
  };

  const handleRunBBO = async () => {
    setIsLoading(true);
    setLoadingText("好みを学習し、量子アニーリングで最適化中...");
    try {
        const res = await api.runBBOIteration({
            gemini_key: state.gemini_key,
            amplify_token: state.amplify_token,
            params: state.params
        });
        if (res.status === 'success') {
            updateState({ 
                candidates: res.candidates, 
                bbo_history_count: res.history_count,
                optimization_plot_data: res.plot_data,
                optimization_scales: res.scales
            });
            alert("最適化完了。最適なブロックがハイライトされました。");
        } else {
            alert("エラー: " + res.message);
        }
    } catch (e) {
        alert("BBO実行中にエラーが発生しました");
    } finally {
        setIsLoading(false);
    }
  };

  const handleRunLegacyOptimization = async () => {
    setIsLoading(true);
    setLoadingText("パラメータで最適化中...");
    try {
        const res = await api.runOptimization({
             amplify_token: state.amplify_token,
             params: state.params
        });
        if (res.status === 'success') {
            updateState({ 
                candidates: res.candidates,
                optimization_plot_data: res.plot_data,
                optimization_scales: res.scales 
            });
        } else {
            alert("エラー: " + res.message);
        }
    } catch (e) {
        alert("エラー");
    } finally {
        setIsLoading(false);
    }
  };

  const handleResetBBO = async () => {
    if (!confirm("学習履歴をリセットしますか？")) return;
    await api.resetBBO();
    updateState({ bbo_history_count: 0 });
    const newCandidates = state.candidates.map(c => ({ ...c, user_rating: 0 }));
    updateState({ candidates: newCandidates });
  };

  const handleGenerateDraft = async () => {
    setIsLoading(true);
    setLoadingText("ドラフト生成中...");
    try {
        const res = await api.generateDraft();
        if (res.status === 'success') {
            updateState({ draft_summary: res.summary, draft_article: res.article });
        } else {
            alert(res.message);
        }
    } catch (e) {
        alert("エラー");
    } finally {
        setIsLoading(false);
    }
  };

  const handleGenerateFinal = async () => {
    await api.saveDraftEdit(state.draft_article, state.additional_instruction);
    setIsLoading(true);
    setLoadingText("最終出力を仕上げ中...");
    try {
        const res = await api.generateFinal();
        if (res.status === 'success') {
            updateState({ final_text: res.final_text });
        } else {
            alert(res.message);
        }
    } catch(e) {
        alert("エラー");
    } finally {
        setIsLoading(false);
    }
  };

  const tabs = [
    { label: "1. 設定 & 生成", id: 0 },
    { label: "2. 最適化 (BBO)", id: 1 },
    { label: "3. ドラフト", id: 2 },
    { label: "4. 最終出力", id: 3 },
  ];

  return (
    <div className="min-h-screen pb-20">
      {isLoading && (
          <div className="fixed inset-0 bg-slate-900/90 z-50 flex flex-col items-center justify-center">
              <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-500 mb-4"></div>
              <h2 className="text-xl text-gray-200 font-semibold">{loadingText}</h2>
          </div>
      )}

      <div className="max-w-5xl mx-auto pt-10 px-4">
        <div className="text-center mb-10">
            <h1 className="text-3xl font-bold text-gray-100 mb-2 flex items-center justify-center">
                <span className="text-blue-500 mr-2">✦</span> 
                Quantum Novel Assistant 
                <span className="ml-3 text-sm bg-blue-600 px-2 py-0.5 rounded-full text-white align-top">Ver.9 React</span>
            </h1>
            <p className="text-gray-400">Human-in-the-Loop ブラックボックス最適化による小説執筆支援</p>
        </div>

        <div className="bg-[#242d3a] rounded-xl shadow-2xl overflow-hidden min-h-[600px]">
            {/* Tabs */}
            <div className="flex border-b border-slate-700 overflow-x-auto">
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`px-6 py-4 font-medium text-sm whitespace-nowrap transition-colors ${
                            activeTab === tab.id 
                            ? 'text-blue-400 border-b-2 border-blue-400 bg-slate-800/50' 
                            : 'text-gray-400 hover:text-gray-200 hover:bg-slate-800/30'
                        }`}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Content */}
            <div className="p-6 md:p-8">
                {activeTab === 0 && (
                    <SettingsTab 
                        state={state} 
                        updateState={updateState} 
                        updateParams={updateParams} 
                        onGenerate={handleGenerateCandidates}
                        isLoading={isLoading}
                    />
                )}
                {activeTab === 1 && (
                    <OptimizationTab 
                        state={state}
                        updateCandidateRating={updateCandidateRating}
                        runBBO={handleRunBBO}
                        runLegacyOptimization={handleRunLegacyOptimization}
                        resetBBO={handleResetBBO}
                        isLoading={isLoading}
                    />
                )}
                {activeTab === 2 && (
                    <DraftTab 
                        state={state}
                        generateDraft={handleGenerateDraft}
                        updateState={updateState}
                        isLoading={isLoading}
                    />
                )}
                {activeTab === 3 && (
                    <FinalTab 
                        state={state}
                        generateFinal={handleGenerateFinal}
                        isLoading={isLoading}
                        updateState={updateState}
                    />
                )}
            </div>
        </div>
      </div>
    </div>
  );
};

export default App;