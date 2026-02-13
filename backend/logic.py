import json
import warnings
import numpy as np
import time
import numbers  # 追加: 数値判定用
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
        if not api_key: raise Exception("APIキーが設定されていません。")
        
        genai.configure(api_key=api_key)
        
        # モデルを gemini-1.5-flash に変更 (2.5は存在しないため1.5へ修正推奨、あるいは最新の有効なモデル)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
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
        if not candidates: return 0  # 候補がない場合は0を返す

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
        if not candidates:
            # 候補がない場合は処理をスキップ
            return [], [], {"pref": 0, "diff": 0, "constraint": 0}

        # ★修正: 制約式が数値(0など)になっていないかチェック。数値の場合はsolveできないためダミーを返す
        if isinstance(model_constraints, numbers.Number) and model_constraints == 0:
             print("Warning: Constraint model is trivial (0). Skipping optimization.")
             return [c.to_dict() for c in candidates], [], {"pref": 0, "diff": 0, "constraint": 0}

        client = FixstarsClient()
        client.token = token
        # ★修正: タイムアウト設定 (ミリ秒単位)
        client.parameters.timeout = 2000 
        
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
        model_step1 = model_constraints
        
        # 数値チェック（Step1用）
        if isinstance(model_step1, numbers.Number):
            print("Step 1 Model is a number. Setting scales to 1.0")
            scales = {key: 1.0 for key in model_objs.keys()}
            values_step1 = None
        else:
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
                    # 目的関数も数値の場合は評価しない
                    if isinstance(obj, numbers.Number):
                        val = abs(obj)
                    else:
                        val = abs(obj.evaluate(values_step1))
                    scales[key] = max(val, 0.1)
            else:
                for key in model_objs.keys(): scales[key] = 1.0
            
        # --- Step 2: Weighted & Normalized Solve ---
        print("\n=== [Step 2] Weighted Optimization ===")
        w_constraint = weights.get('constraint', 1.0)
        
        # ★修正: モデルの初期化を慎重に行う
        model_final = model_constraints * w_constraint
        
        for key, obj in model_objs.items():
            s = scales.get(key, 1.0)
            w = weights.get(key, 1.0)
            model_final += (obj / s) * w
            print(f"{key}: Scale = {s:.4f}, Weight = {w}")
            
        # ★修正: 最終モデルが数値の場合はsolveを実行しない
        if isinstance(model_final, numbers.Number):
            print("Final Model is a number (trivial). Skipping solve.")
            values = values_step1
            plot_data = []
        else:
            client.parameters.timeout = 5000 # 5秒
            result = solve(model_final, client)
            
            # --- Extract Solutions & Plot Data ---
            plot_data = []
            solutions = []
            # ソリューションの取得方法を網羅
            if hasattr(result, 'solutions'): solutions = result.solutions
            elif isinstance(result, list): solutions = result
            else: solutions = [result]

            for sol in solutions:
                t = getattr(sol, 'time', 0.0)
                # v1の場合はtotal_seconds()が必要な場合がある
                if hasattr(t, 'total_seconds'): t = t.total_seconds()
                
                v = getattr(sol, 'energy', 0)
                # 値がComplexや特殊な型の場合の変換
                if hasattr(v, 'real'): v = v.real
                
                plot_data.append({"time": float(t), "value": float(v)})
            
            plot_data.sort(key=lambda x: x['time'])

            # グラフが見やすいようにデータを補間・整形
            if plot_data:
                # 0秒地点を追加（初期状態）
                if plot_data[0]['time'] > 0:
                    plot_data.insert(0, {"time": 0.0, "value": plot_data[0]['value'] * 1.1})
                
                # タイムアウト時間までデータを伸ばす（最後の値を維持）
                last_time = plot_data[-1]['time']
                timeout_sec = client.parameters.timeout / 1000.0
                if last_time < timeout_sec:
                    plot_data.append({"time": timeout_sec, "value": plot_data[-1]['value']})

            # Apply best solution
            values = None
            if hasattr(result, 'best'): values = result.best.values
            elif isinstance(result, list) and len(result) > 0: values = result[0].values
        
        # Fallback logic
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
        
        print("\n=== Final Results ===")
        if values is not None:
            for key, obj in model_objs.items():
                try:
                    raw_val = obj.evaluate(values) if not isinstance(obj, numbers.Number) else obj
                    print(f"{key} (Raw Value): {raw_val:.4f}")
                except: pass
        
        for i, c in enumerate(candidates):
            val = get_bit_value(values, q[i], i)
            c.selected = (val > 0.5)
            if c.selected:
                final_selected_len += len(c.text)
            updated_candidates.append(c.to_dict())
        
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
        # q[i]がPolyであることを前提に計算するが、候補0の場合は0になる
        if not candidates:
             current_len = 0
        else:
             current_len = sum([len(c.text) * q[i] for i, c in enumerate(candidates)])
             
        target = float(params['length'])
        h_len_penalty = 0.001 * (current_len - target)**2
        
        weights = {
            "diff": 1.0,
            "constraint": 1.0
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
        
        if not candidates:
            current_len = 0
        else:
            current_len = sum([len(c.text) * q[i] for i, c in enumerate(candidates)])
            
        target = float(params['length'])
        h_len_penalty = 0.001 * (current_len - target)**2
        
        weights = {
            "pref": 10.0,
            "diff": 1.0,
            "constraint": 1.0
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
        if not api_key: raise Exception("APIキーが設定されていません。")
        
        genai.configure(api_key=api_key)
        
        # モデルを gemini-1.5-flash に変更
        model = genai.GenerativeModel("gemini-1.5-flash")
        
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
        if not api_key: raise Exception("APIキーが設定されていません。")
        
        genai.configure(api_key=api_key)
        
        # モデルを gemini-1.5-flash に変更
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        return LogicHandler._safe_generate(model, f"以下の指示に基づいてこの文章を推敲してください: {instr}\n文章:\n{draft}")