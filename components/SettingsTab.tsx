import React, { useRef } from 'react';
import { AppState, Params } from '../types';
import { Upload, Download, FolderOpen, Key, Book, Palette, Users, Zap } from 'lucide-react';

interface Props {
  state: AppState;
  updateState: (updates: Partial<AppState>) => void;
  updateParams: (updates: Partial<Params>) => void;
  onGenerate: () => void;
  isLoading: boolean;
}

export const SettingsTab: React.FC<Props> = ({ state, updateState, updateParams, onGenerate, isLoading }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 【重要修正】サーバー通信を行わず、画面のstateを直接JSONファイル化してダウンロードする
  const handleDownloadSettings = () => {
    // 1. 現在の画面の状態(state)をJSONテキストに変換
    const jsonString = JSON.stringify(state, null, 2);
    
    // 2. ブラウザ内でファイルデータ(Blob)を作成
    const blob = new Blob([jsonString], { type: 'application/json' });
    
    // 3. 一時的なダウンロードリンクを作成してクリックさせる
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'settings.json';
    document.body.appendChild(link);
    link.click();
    
    // 4. 後始末
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
       const formData = new FormData();
       formData.append('file', e.target.files[0]);
       try {
           // アップロード処理はサーバー側で行い、解析結果を受け取る
           const res = await fetch('/api/settings/upload', { method: 'POST', body: formData });
           
           const contentType = res.headers.get("content-type");
           if (contentType && contentType.indexOf("application/json") !== -1) {
               const result = await res.json();
               if (result.status === 'success' && result.settings) {
                   const s = result.settings;
                   updateState({
                       gemini_key: s.gemini_key || '',
                       amplify_token: s.amplify_token || '',
                       topic_main: s.topic_main || '',
                       topic_sub1: s.topic_sub1 || '',
                       topic_sub2: s.topic_sub2 || '',
                       candidates: s.candidates || [],
                       bbo_history_count: s.bbo_history ? s.bbo_history.length : 0,
                       draft_summary: s.draft_summary || '',
                       draft_article: s.draft_article || '',
                       additional_instruction: s.additional_instruction || '',
                       final_text: s.final_text || ''
                   });
                   if (s.params) updateParams(s.params);
                   alert("設定ファイルを読み込み、反映しました。");
               } else {
                   alert("読み込みエラー: " + (result.message || "不明なエラー"));
               }
           } else {
               const text = await res.text();
               console.error("Server Error Response:", text);
               alert("サーバーエラーが発生しました（JSON以外の応答）。");
           }
       } catch (err) {
           console.error(err);
           alert("通信エラーが発生しました: " + (err as Error).message);
       }
       
       if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const renderSlider = (
    labelLeft: string,
    labelRight: string,
    value: number,
    field: keyof Params,
    step: number = 0.1,
    min: number = 0,
    max: number = 1
  ) => (
    <div className="mb-4">
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{labelLeft}</span>
        <span>{labelRight}</span>
      </div>
      <input
        type="range"
        className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-400"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => updateParams({ [field]: parseFloat(e.target.value) })}
      />
    </div>
  );

  return (
    <div className="animate-fadeIn">
      {/* API Keys */}
      <div className="mb-6">
        <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <h3 className="text-gray-200 font-bold mb-3 flex items-center">
                <Key className="w-4 h-4 mr-2 text-blue-400" />
                APIキー設定
            </h3>
            <div className="space-y-3">
                <div>
                    <label className="text-xs text-gray-400 block mb-1">Gemini API キー</label>
                    <input 
                        type="password" 
                        className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
                        placeholder="Gemini APIキーを入力"
                        value={state.gemini_key}
                        onChange={(e) => updateState({ gemini_key: e.target.value })}
                    />
                </div>
                <div>
                    <label className="text-xs text-gray-400 block mb-1">Fixstars Amplify トークン</label>
                    <input 
                        type="password" 
                        className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
                        placeholder="Amplifyトークンを入力"
                        value={state.amplify_token}
                        onChange={(e) => updateState({ amplify_token: e.target.value })}
                    />
                </div>
            </div>
        </div>
      </div>

      {/* Settings File */}
      <div className="mb-6 bg-slate-800 rounded-lg p-4 border border-slate-700">
        <h6 className="text-blue-200 font-semibold mb-2 flex items-center">
            <FolderOpen className="w-4 h-4 mr-2" />
            設定ファイル (settings.json)
        </h6>
        <p className="text-xs text-gray-400 mb-4">
            現在の設定をダウンロードして保存するか、アップロードして復元します。
        </p>
        <div className="flex flex-wrap gap-4">
            {/* 【重要修正】ここが <a> タグではなく、onClickイベントを持つ <button> である必要があります */}
            <button 
                onClick={handleDownloadSettings}
                className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm text-white transition cursor-pointer"
            >
                <Download className="w-4 h-4" /> ダウンロード
            </button>
            <div className="flex items-center gap-2 bg-blue-900/20 px-4 py-2 rounded-lg border border-blue-500/20">
                <label className="cursor-pointer flex items-center gap-2 text-sm text-blue-400 hover:text-blue-300 transition">
                    <Upload className="w-4 h-4" />
                    <span>アップロード</span>
                    <input 
                        ref={fileInputRef} 
                        type="file" 
                        accept=".json" 
                        className="hidden" 
                        onChange={handleFileUpload} 
                    />
                </label>
            </div>
        </div>
      </div>

      {/* Scene Settings */}
      <div className="mb-8">
        <h5 className="text-lg font-bold text-gray-200 mb-4 flex items-center">
            <Book className="w-5 h-5 mr-2 text-blue-400" />
            シーン設定
        </h5>
        <div className="space-y-4">
            <div>
                <label className="block text-sm font-bold mb-1 text-gray-300">設定 1 (必須) <span className="text-red-400">*</span></label>
                <textarea 
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg p-3 text-sm text-gray-200 focus:border-blue-500 focus:outline-none min-h-[80px]"
                    placeholder="例: 主人公は廃墟となった遊園地で失われた記憶の手がかりを探している。"
                    value={state.topic_main}
                    onChange={(e) => updateState({ topic_main: e.target.value })}
                />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label className="block text-sm font-bold mb-1 text-gray-300">設定 2 (任意)</label>
                    <textarea 
                        className="w-full bg-slate-800 border border-slate-700 rounded-lg p-3 text-sm text-gray-200 focus:border-blue-500 focus:outline-none min-h-[80px]"
                        placeholder="例: 激しい雨が降っており、視界が悪い。"
                        value={state.topic_sub1}
                        onChange={(e) => updateState({ topic_sub1: e.target.value })}
                    />
                </div>
                <div>
                    <label className="block text-sm font-bold mb-1 text-gray-300">設定 3 (任意)</label>
                    <textarea 
                        className="w-full bg-slate-800 border border-slate-700 rounded-lg p-3 text-sm text-gray-200 focus:border-blue-500 focus:outline-none min-h-[80px]"
                        placeholder="例: 主人公は古い懐中時計を握りしめている。"
                        value={state.topic_sub2}
                        onChange={(e) => updateState({ topic_sub2: e.target.value })}
                    />
                </div>
            </div>
        </div>
      </div>

      <hr className="border-slate-700 my-6" />

      {/* Sliders */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
            <h6 className="text-blue-400 border-b border-slate-700 pb-2 mb-4 flex items-center font-semibold">
                <Palette className="w-4 h-4 mr-2" /> シーン描写 (Scene Craft)
            </h6>
            {renderSlider("説明的", "描写的", state.params.p_desc_style, 'p_desc_style')}
            {renderSlider("客観的", "没入的", state.params.p_perspective, 'p_perspective')}
            {renderSlider("感覚描写 少", "感覚描写 多", state.params.p_sensory, 'p_sensory')}
            {renderSlider("内面隠蔽", "内面吐露", state.params.p_thought, 'p_thought')}
            {renderSlider("弛緩", "緊張", state.params.p_tension, 'p_tension')}
            {renderSlider("現実感（低）", "現実感（高）", state.params.p_reality, 'p_reality')}
        </div>
        <div>
            <h6 className="text-blue-400 border-b border-slate-700 pb-2 mb-4 flex items-center font-semibold">
                <Users className="w-4 h-4 mr-2" /> キャラクター造形 (Character Dynamics)
            </h6>
            {renderSlider("0人", "5人", state.params.p_char_count, 'p_char_count', 0.2)}
            {renderSlider("心理描写（弱）", "心理描写（強）", state.params.p_char_mental, 'p_char_mental')}
            {renderSlider("信念（弱）", "信念（強）", state.params.p_char_belief, 'p_char_belief')}
            {renderSlider("トラウマ無", "トラウマ有", state.params.p_char_trauma, 'p_char_trauma')}
            {renderSlider("主張（弱）", "主張（強）", state.params.p_char_voice, 'p_char_voice')}
            
            <div className="mt-6 pt-4 border-t border-slate-700">
                <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-bold text-gray-300">目標文字数</span>
                    <span className="px-2 py-1 bg-slate-700 rounded text-xs text-white">{state.params.length} 文字</span>
                </div>
                <input
                    type="range"
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-400"
                    min={200}
                    max={1000}
                    step={50}
                    value={state.params.length}
                    onChange={(e) => updateParams({ length: parseInt(e.target.value) })}
                />
            </div>
        </div>
      </div>

      <div className="mt-8 text-center">
        <button 
            onClick={onGenerate}
            disabled={isLoading}
            className="bg-gradient-to-r from-blue-400 to-blue-600 hover:from-blue-500 hover:to-blue-700 text-white font-bold py-3 px-8 rounded-full shadow-lg shadow-blue-500/30 transform transition hover:-translate-y-1 disabled:opacity-50 disabled:cursor-not-allowed flex items-center mx-auto"
        >
            {isLoading ? (
                <span className="animate-spin mr-2">⟳</span>
            ) : (
                <Zap className="w-5 h-5 mr-2" />
            )}
            候補を生成 (各８個)
        </button>
      </div>
    </div>
  );
};
