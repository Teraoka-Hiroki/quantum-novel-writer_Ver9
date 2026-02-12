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
    def generate_candidates_api(api_key, topic_main, topic_sub1, topic_sub2, params):
        if not HAS_GENAI: raise Exception("gemini lib missing")
        
        # 【修正】APIキーを設定（引数名 api_key を使用）
        genai.configure(api_key=api_key)
        
        # 【修正】モデルを gemini-1.5-flash に変更
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""
        以下の設定に基づいて、30個の執筆ブロック（「Scene Craft」（シーン描写）15個、「Character Dynamics」（キャラクター造形）15個）を日本語で生成してください。
        
        メイン設定: {topic_main}
        サブ設定1: {topic_sub1}
        サブ設定2: {topic_sub2}
        
        以下の有効なJSONリストのみを返してください。
        "type"は厳密に "Scene Craft" または "Character Dynamics" としてください。
        
        [
          {{ "type": "Scene Craft", "text": "...", "scores": {{ "relevance": 0.5, "desc_style": 0.5, "perspective": 0.5, "sensory": 0.5, "thought": 0.5, "tension": 0.5, "reality": 0.5 }} }},
          {{ "type": "Character Dynamics", "text": "...", "scores": {{ "relevance": 0.5, "char_count": 0.2, "char_mental": 0.5, "char_belief": 0.5, "char_trauma": 0.0, "char_voice": 0.5 }} }}
        ]
        """
        text = LogicHandler._safe_generate(model, prompt).replace("```json", "").replace("```", "").strip()
        start, end = text.find("["), text.rfind("]")
        data = json.loads(text[start:end+1])
        return [DraftItem(i, item["text"], item["type"], item["scores"].get("relevance", 0.5), item["scores"]) for i, item in enumerate(data)]

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
        """
        Mult-stage optimization with customizable weights and console logging.
        """
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
        
        # --- Step 1: Constraint Only to get magnitude (Approximation) ---
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
            # Calculate rough current length for logging
            approx_len = sum([len(c.text) for i, c in enumerate(candidates) if get_bit_value(values_step1, q[i], i) > 0.5])
            print(f"Step1 Approx Length: {approx_len} (Target: {target_length})")
            
            for key, obj in model_objs.items():
                val = abs(obj.evaluate(values_step1))
                scales[key] = max(val, 0.1) # Avoid zero division
        else:
            for key in model_objs.keys(): scales[key] = 1.0
            
        # --- Step 2: Weighted & Normalized Solve ---
        print("\n=== [Step 2] Weighted Optimization ===")
        # ① 重み係数の適用
        w_constraint = weights.get('constraint', 1.0)
        model_final = model_constraints * w_constraint
        print(f"Constraint: Weight = {w_constraint}")

        for key, obj in model_objs.items():
            s = scales[key]
            w = weights.get(key, 1.0)
            # Normalize (obj/s) then apply weight (w)
            model_final += (obj / s) * w
            print(f"{key}: Scale = {s:.4f}, Weight = {w}")
            
        client.parameters.timeout = 4000
        result = solve(model_final, client)
        
        # --- Extract Solutions & Plot Data ---
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

        # Apply best solution
        values = None
        if hasattr(result, 'best'): values = result.best.values
        elif isinstance(result, list) and len(result) > 0: values = result[0].values
        
        # Fallback
        step2_has_selection = False
        if values is not None:
            for i in range(len(candidates)):
                if get_bit_value(values, q[i], i) > 0.5:
                    step2_has_selection = True
                    break
        if not step2_has_selection and values_step1 is not None:
             values = values_step1
             print("Step 2 yield no selection. Reverting to Step 1 results.")
        
        # Update Candidates & Log Final Values
        updated_candidates = []
        final_selected_len = 0
        
        # ② コンソール出力（最終値）
        print("\n=== Final Results ===")
        if values is not None:
            # Calculate raw values for logging
            for key, obj in model_objs.items():
                try:
                    raw_val = obj.evaluate(values)
                    print(f"{key} (Raw Value): {raw_val:.4f}")
                except: pass
        
        for i, c in enumerate(candidates):
            val = get_bit_value(values, q[i], i)
            c.selected = (val > 0.5)
            if c.selected:
                final_selected_len += len(c.text)
            updated_candidates.append(c.to_dict())
        
        # Calculate constraint value
        target_val = float(target_length) if target_length else 500.0
        constraint_val = 0.001 * (final_selected_len - target_val)**2
        print(f"Constraint (Raw Value): {constraint_val:.4f}")
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
        
        # ③ 推奨係数の設定（パラメータ最適化のみの場合）
        weights = {
            "diff": 1.0,       # 設定への適合を重視
            "constraint": 1.0  # 制約は標準〜緩め
        }
        
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
            # Minimize negative preference (Maximize preference)
            h_user_pref -= predicted[i] * q[i]
            
        h_param_diff = LogicHandler._construct_param_objective(q, candidates, params)
        current_len = sum([len(c.text) * q[i] for i, c in enumerate(candidates)])
        target = float(params['length'])
        h_len_penalty = 0.001 * (current_len - target)**2
        
        # ③ 推奨係数の設定（BBOの場合）
        weights = {
            "pref": 10.0,       # 好みを最優先
            "diff": 1.0,       # 設定も重視
            "constraint": 1.0  # 制約は緩く
        }
        
        return LogicHandler._solve_multi_stage(
            token,
            {"pref": h_user_pref, "diff": h_param_diff},
            h_len_penalty,
            q, candidates, target, weights
        )

    @staticmethod
    def generate_draft(api_key, selected, params):
        if not HAS_GENAI: raise Exception("gemini lib missing")
        
        # 【修正】APIキーを設定（引数名 api_key を使用）
        genai.configure(api_key=api_key)
        
        # 【修正】モデルを gemini-1.5-flash に変更
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
        
        # 【修正】APIキーを設定（引数名 api_key を使用）
        genai.configure(api_key=api_key)
        
        # 【修正】モデルを gemini-1.5-flash に変更
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        return LogicHandler._safe_generate(model, f"以下の指示に基づいてこの文章を推敲してください: {instr}\n文章:\n{draft}")