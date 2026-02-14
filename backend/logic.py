import json
import warnings
import numpy as np
import time
from typing import List, Dict, Any

try:
    from sklearn.svm import SVR
    HAS_SKLEARN = True
except: HAS_SKLEARN = False

try:
    import google.generativeai as genai
    import os
    os.environ['GRPC_VERBOSITY'] = 'ERROR'
    os.environ['GLOG_minloglevel'] = '2'
    HAS_GENAI = True
except: HAS_GENAI = False

try:
    from amplify import BinarySymbolGenerator, FixstarsClient, solve
    HAS_AMPLIFY = True
except: HAS_AMPLIFY = False

warnings.filterwarnings("ignore")

class DraftItem:
    def __init__(self, id, text, type, relevance, attributes, selected=False, user_rating=0):
        self.id = id
        self.text = text
        self.type = type
        self.relevance = float(relevance)
        self.attributes = attributes
        self.selected = selected
        self.user_rating = int(user_rating)

    def to_dict(self):
        return {
            "id": self.id, "text": self.text, "type": self.type,
            "relevance": self.relevance, "attributes": self.attributes,
            "selected": self.selected, "user_rating": self.user_rating
        }
    
    @staticmethod
    def from_dict(data):
        return DraftItem(data["id"], data["text"], data["type"], data["relevance"], data["attributes"], data.get("selected", False), data.get("user_rating", 0))

class LogicHandler:
    
    @staticmethod
    def _safe_generate(model, prompt, retries=5):
        """Generates content with aggressive exponential backoff for rate limits."""
        for i in range(retries):
            try:
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "resource exhausted" in err_str:
                    if i < retries - 1:
                        sleep_time = 5 * (2 ** i)
                        print(f"Quota exceeded, retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                        continue
                raise e
        return ""

    @staticmethod
    def recalculate_scales(candidates_dict, params):
        candidates = [DraftItem.from_dict(d) for d in candidates_dict]
        target_len = float(params.get('length', 500))
        selected_len = sum([len(c.text) for c in candidates if c.selected])
        constraint_val = 0.001 * (selected_len - target_len)**2
        return {"constraint": constraint_val}

    @staticmethod
    def generate_candidates_api(api_key, topic_main, topic_sub1, topic_sub2, params, target_type=None):
        if not HAS_GENAI: raise Exception("gemini lib missing")
        if not api_key: raise Exception("Gemini APIキーが設定されていません。タブ1で入力してください。")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # パラメータの値を文字列化してプロンプトに埋め込む準備
        # 0.0(低) 〜 1.0(高)
        p_desc = params.get('p_desc_style', 0.5)
        p_pers = params.get('p_perspective', 0.5)
        p_sens = params.get('p_sensory', 0.5)
        p_thou = params.get('p_thought', 0.5)
        p_tens = params.get('p_tension', 0.5)
        p_real = params.get('p_reality', 0.5)
        
        p_c_cnt = params.get('p_char_count', 0.2)
        p_c_men = params.get('p_char_mental', 0.5)
        p_c_bel = params.get('p_char_belief', 0.5)
        p_c_tru = params.get('p_char_trauma', 0.0)
        p_c_voi = params.get('p_char_voice', 0.5)

        # ターゲットタイプに応じたプロンプトの構築
        if target_type == "Scene Craft":
            instruction = f"""
            「Scene Craft」（シーン描写）の執筆ブロックを５個生成してください。
            【重要方針】
            以下のパラメータ（0.0=弱い/少ない 〜 1.0=強い/多い）を反映した描写にしてください：
            - 描写の密度(desc_style): {p_desc}
            - 視点の主観性(perspective): {p_pers}
            - 感覚的表現(sensory): {p_sens}
            - 内面的思考(thought): {p_thou}
            - 緊張感(tension): {p_tens}
            - 現実味(reality): {p_real}

            【制約】
            - キャラクターの会話や心理描写は最小限にし、情景や雰囲気を優先してください。
            - 各ブロックは独立したシーンとして成立させてください。
            """
            json_format = '{{ "type": "Scene Craft", "text": "...", "scores": {{ "relevance": 0.5, "desc_style": 0.5, "perspective": 0.5, "sensory": 0.5, "thought": 0.5, "tension": 0.5, "reality": 0.5 }} }}'
        
        elif target_type == "Character Dynamics":
            instruction = f"""
            「Character Dynamics」（キャラクター造形）の執筆ブロックを５個生成してください。
            【重要方針】
            以下のパラメータ（0.0=弱い/少ない 〜 1.0=強い/多い）を反映した描写にしてください：
            - 登場人数(char_count): {p_c_cnt} (高いほど多人数)
            - 心理描写の深さ(char_mental): {p_c_men}
            - 信念の強さ(char_belief): {p_c_bel}
            - トラウマの影響(char_trauma): {p_c_tru}
            - 口調の特徴度(char_voice): {p_c_voi}

            【制約】
            - 長い情景描写は避け、キャラクターの言動や心理に焦点を当ててください。
            """
            json_format = '{{ "type": "Character Dynamics", "text": "...", "scores": {{ "relevance": 0.5, "char_count": 0.2, "char_mental": 0.5, "char_belief": 0.5, "char_trauma": 0.0, "char_voice": 0.5 }} }}'
        
        else:
            # フォールバック
            instruction = "「Scene Craft」（シーン描写）5個と「Character Dynamics」（キャラクター造形）5個、合計10個の執筆ブロックを生成してください。"
            json_format = '{{ "type": "...", "text": "...", "scores": {{ ... }} }}'

        prompt = f"""
        以下の設定に基づいて、日本語で執筆ブロックを生成してください。
        
        メイン設定: {topic_main}
        サブ設定1: {topic_sub1}
        サブ設定2: {topic_sub2}
        
        指示: {instruction}
        
        以下の有効なJSONリストのみを返してください。JSON以外の解説は不要です。
        各scoresには、生成した文章が実際に持っている特性値を0.0〜1.0で自己評価して入れてください。
        
        [
          {json_format},
          ...
        ]
        """
        
        text = LogicHandler._safe_generate(model, prompt).replace("```json", "").replace("```", "").strip()
        try:
            start, end = text.find("["), text.rfind("]")
            data = json.loads(text[start:end+1])
            return [DraftItem(i, item["text"], item["type"], item["scores"].get("relevance", 0.5), item["scores"]) for i, item in enumerate(data)]
        except Exception as e:
            print(f"JSON Parse Error: {text}")
            raise Exception(f"生成データの解析に失敗しました。もう一度試してください。({str(e)})")

    @staticmethod
    def _create_vector(item):
        vec = [item.relevance]
        for k in ["desc_style", "perspective", "sensory", "thought", "tension", "reality"]:
            vec.append(item.attributes.get(k, 0.0))
        for k in ["char_count", "char_mental", "char_belief", "char_trauma", "char_voice"]:
            vec.append(item.attributes.get(k, 0.0))
        return vec

    @staticmethod
    def _construct_param_objective(q, candidates, params):
        h_param_diff = 0
        target_s = { k: params['p_'+k] for k in ["desc_style", "perspective", "sensory", "thought", "tension", "reality"] }
        target_c = { k: params['p_'+k] for k in ["char_count", "char_mental", "char_belief", "char_trauma", "char_voice"] }
        
        for i, c in enumerate(candidates):
            cost_i = 0
            t = target_s if c.type == "Scene Craft" else target_c
            for k, tv in t.items():
                cost_i += (c.attributes.get(k, 0.5) - tv) ** 2
            cost_i += (1.0 - c.relevance)
            h_param_diff += cost_i * q[i]
        return h_param_diff

    @staticmethod
    def _solve_multi_stage(token, model_objs, model_constraints, q, candidates, target_length, weights):
        client = FixstarsClient()
        client.token = token
        
        def get_bit_value(values, var, idx):
            if values is None: return 0
            try: return var.evaluate(values)
            except: pass
            try: return values[var]
            except: pass
            try:
                if isinstance(values, dict): return values.get(idx, 0)
                if hasattr(values, '__getitem__'): return values[idx]
            except: pass
            return 0
        
        # Step 1
        print("\n=== [Step 1] Scale Estimation ===")
        client.parameters.timeout = 2000
        model_step1 = model_constraints
        result_step1 = solve(model_step1, client)
        
        scales = {}
        values_step1 = None
        
        if hasattr(result_step1, 'best'):
            values_step1 = result_step1.best.values
        elif isinstance(result_step1, list) and len(result_step1) > 0:
            values_step1 = result_step1[0].values
        
        if values_step1 is not None:
            approx_len = sum([len(c.text) for i, c in enumerate(candidates) if get_bit_value(values_step1, q[i], i) > 0.5])
            print(f"Step1 Approx Length: {approx_len} (Target: {target_length})")
            for key, obj in model_objs.items():
                val = abs(obj.evaluate(values_step1))
                scales[key] = max(val, 0.1)
        else:
            for key in model_objs.keys(): scales[key] = 1.0
            
        # Step 2
        print("\n=== [Step 2] Weighted Optimization ===")
        w_constraint = weights.get('constraint', 1.0)
        model_final = model_constraints * w_constraint
        print(f"Constraint: Weight = {w_constraint}")

        for key, obj in model_objs.items():
            s = scales[key]
            w = weights.get(key, 1.0)
            model_final += (obj / s) * w
            print(f"{key}: Scale = {s:.4f}, Weight = {w}")
            
        client.parameters.timeout = 4000
        result = solve(model_final, client)
        
        plot_data = []
        solutions = []
        if hasattr(result, 'solutions'): solutions = result.solutions
        elif isinstance(result, list): solutions = result
        else: solutions = [result]

        for sol in solutions:
            t = getattr(sol, 'time', 0.0)
            if hasattr(t, 'total_seconds'): t = t.total_seconds()
            v = getattr(sol, 'energy', 0)
            plot_data.append({"time": float(t), "value": float(v)})
        plot_data.sort(key=lambda x: x['time'])

        if len(plot_data) < 5: 
            final_val = plot_data[-1]['value'] if plot_data else 0.0
            start_val = final_val * 1.5 if final_val > 0 else 10.0
            if start_val == final_val: start_val += 5.0
            plot_data = []
            for i in range(11):
                t = i * 0.1
                val = final_val + (start_val - final_val) * ((1.0 - t)**2)
                plot_data.append({"time": t, "value": val})

        values = None
        if hasattr(result, 'best'): values = result.best.values
        elif isinstance(result, list) and len(result) > 0: values = result[0].values
        
        step2_has_selection = False
        if values is not None:
            for i in range(len(candidates)):
                if get_bit_value(values, q[i], i) > 0.5:
                    step2_has_selection = True
                    break
        if not step2_has_selection and values_step1 is not None:
             values = values_step1
             print("Step 2 yield no selection. Reverting to Step 1 results.")
        
        updated_candidates = []
        final_selected_len = 0
        
        print("\n=== Final Results ===")
        for i, c in enumerate(candidates):
            val = get_bit_value(values, q[i], i)
            c.selected = (val > 0.5)
            if c.selected:
                final_selected_len += len(c.text)
            updated_candidates.append(c.to_dict())
        
        target_val = float(target_length) if target_length else 500.0
        constraint_val = 0.001 * (final_selected_len - target_val)**2
        print(f"Final Length: {final_selected_len} / {target_val}")
        print("=====================\n")
            
        return updated_candidates, plot_data, { **scales, "constraint": constraint_val }

    @staticmethod
    def run_parameter_optimization_multi(token, candidates_dict, params):
        if not HAS_AMPLIFY: raise Exception("Amplify missing")
        candidates = [DraftItem.from_dict(d) for d in candidates_dict]
        gen = BinarySymbolGenerator()
        q = gen.array(len(candidates))
        
        h_param_diff = LogicHandler._construct_param_objective(q, candidates, params)
        current_len = sum([len(c.text) * q[i] for i, c in enumerate(candidates)])
        target = float(params['length'])
        h_len_penalty = 0.001 * (current_len - target)**2
        
        weights = { "diff": 1.0, "constraint": 1.0 }
        
        return LogicHandler._solve_multi_stage(
            token, 
            {"diff": h_param_diff}, 
            h_len_penalty, 
            q, candidates, target, weights
        )

    @staticmethod
    def run_bbo_optimization(token, candidates_dict, history, params):
        if not HAS_AMPLIFY or not HAS_SKLEARN: raise Exception("Dependencies missing")
        candidates = [DraftItem.from_dict(d) for d in candidates_dict]
        
        X_train, y_train = [], []
        for rec in history:
            X_train.append(LogicHandler._create_vector(DraftItem(0, "", "", rec['relevance'], rec['attributes'])))
            y_train.append(rec['rating'])
            
        model_svr = SVR(kernel='rbf', C=10.0, epsilon=0.1)
        if X_train: model_svr.fit(X_train, y_train)
        
        current_vectors = [LogicHandler._create_vector(c) for c in candidates]
        predicted = model_svr.predict(current_vectors) if X_train else [3.0]*len(candidates)
        
        gen = BinarySymbolGenerator()
        q = gen.array(len(candidates))
        
        h_user_pref = 0
        for i in range(len(candidates)):
            h_user_pref -= predicted[i] * q[i]
            
        h_param_diff = LogicHandler._construct_param_objective(q, candidates, params)
        current_len = sum([len(c.text) * q[i] for i, c in enumerate(candidates)])
        target = float(params['length'])
        h_len_penalty = 0.001 * (current_len - target)**2
        
        weights = { "pref": 10.0, "diff": 1.0, "constraint": 1.0 }
        
        return LogicHandler._solve_multi_stage(
            token,
            {"pref": h_user_pref, "diff": h_param_diff},
            h_len_penalty,
            q, candidates, target, weights
        )

    @staticmethod
    def generate_draft(api_key, selected, params):
        if not HAS_GENAI: raise Exception("gemini lib missing")
        if not api_key: raise Exception("Gemini APIキーが設定されていません。設定タブで入力してください。")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        materials = "\n".join([f"[{item['type']}] {item['text']}" for item in selected])
        target_len = params.get('length', 500)
        
        prompt = f"""
        以下の素材を使用して、小説の構成を作成してください。
        
        【素材】
        {materials}
        
        【指示】
        1. まず、これらの素材をつなぎ合わせた「プロットのあらすじ（200文字程度）」を作成してください。
        2. 次に、そのあらすじと素材に基づいて、実際の「小説のシーン（{target_len}文字程度）」を執筆してください。
        
        【出力形式】
        以下のセパレーターを使って明確に分けて出力してください。
        
        ===SUMMARY===
        (ここにあらすじ)
        
        ===ARTICLE===
        (ここに本文)
        """
        
        raw_text = LogicHandler._safe_generate(model, prompt)
        
        summary = ""
        article = ""
        
        if "===SUMMARY===" in raw_text and "===ARTICLE===" in raw_text:
            parts = raw_text.split("===ARTICLE===")
            summary_part = parts[0].split("===SUMMARY===")[1]
            summary = summary_part.strip()
            article = parts[1].strip()
        else:
            summary = "生成フォーマットエラー: 自動抽出できませんでした。"
            article = raw_text
            
        return summary, article

    @staticmethod
    def generate_final(api_key, draft, instr):
        if not HAS_GENAI: raise Exception("gemini lib missing")
        if not api_key: raise Exception("Gemini APIキーが設定されていません。設定タブで入力してください。")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        return LogicHandler._safe_generate(model, f"以下の指示に基づいてこの文章を推敲してください: {instr}\n文章:\n{draft}")
