import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_file, send_from_directory, make_response
from logic import LogicHandler
import io
import json
import traceback

app = Flask(__name__, static_folder='../dist', static_url_path='/')

SETTINGS_FILE = "settings.json"

def _default_settings():
    return {
        "gemini_key": os.environ.get("GEMINI_API_KEY", ""),
        "amplify_token": os.environ.get("AMPLIFY_TOKEN", ""),
        "topic_main": "",
        "topic_sub1": "",
        "topic_sub2": "",
        "params": {
            "p_desc_style": 0.5, "p_perspective": 0.5, "p_sensory": 0.5,
            "p_thought": 0.5, "p_tension": 0.5, "p_reality": 0.5,
            "p_char_count": 0.2, "p_char_mental": 0.5, "p_char_belief": 0.5,
            "p_char_trauma": 0.0, "p_char_voice": 0.5, "length": 500
        },
        "candidates": [],
        "draft_summary": "",
        "draft_article": "",
        "additional_instruction": "",
        "final_text": "",
        "bbo_history": [],
        "optimization_scales": {"pref": 0, "diff": 0, "constraint": 0}
    }

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                default = _default_settings()
                for k, v in default.items():
                    if k not in data: data[k] = v
                return data
        except: return _default_settings()
    return _default_settings()

def save_settings(data):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except: pass

DATA_STORE = load_settings()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/generate_candidates', methods=['POST'])
def generate_candidates():
    global DATA_STORE
    req = request.json
    
    # パラメータ保存
    DATA_STORE.update({
        'gemini_key': req.get('gemini_key'),
        'amplify_token': req.get('amplify_token'),
        'topic_main': req.get('topic_main'),
        'topic_sub1': req.get('topic_sub1'),
        'topic_sub2': req.get('topic_sub2'),
        'params': req.get('params'),
    })
    
    # 追記モードフラグとターゲットタイプ
    is_append = req.get('append', False)
    target_type = req.get('target_type', None)
    
    # append=Falseならリセット
    if not is_append:
        DATA_STORE['candidates'] = []
        DATA_STORE['bbo_history'] = []
        
    save_settings(DATA_STORE)
    
    try:
        new_candidates_objs = LogicHandler.generate_candidates_api(
            DATA_STORE['gemini_key'],
            DATA_STORE['topic_main'],
            DATA_STORE['topic_sub1'],
            DATA_STORE['topic_sub2'],
            DATA_STORE['params'],
            target_type=target_type # タイプ指定を渡す
        )
        
        # IDの再採番（追記時の重複防止）
        start_id = 0
        if DATA_STORE['candidates']:
            start_id = max(c['id'] for c in DATA_STORE['candidates']) + 1
            
        for i, c in enumerate(new_candidates_objs):
            c.id = start_id + i
            DATA_STORE['candidates'].append(c.to_dict())
            
        save_settings(DATA_STORE)
        return jsonify({"status": "success", "candidates": DATA_STORE['candidates']})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/update_rating', methods=['POST'])
def update_rating():
    global DATA_STORE
    req = request.json
    item_id = req.get('id')
    rating = req.get('rating')
    for c in DATA_STORE['candidates']:
        if c['id'] == item_id:
            c['user_rating'] = int(rating)
            break
    save_settings(DATA_STORE)
    return jsonify({"status": "success"})

@app.route('/api/optimize', methods=['POST'])
def optimize():
    global DATA_STORE
    req = request.json
    if req.get('amplify_token'): DATA_STORE['amplify_token'] = req.get('amplify_token')
    if req.get('params'): DATA_STORE['params'] = req.get('params')
    save_settings(DATA_STORE)

    try:
        result = LogicHandler.run_parameter_optimization_multi(
            DATA_STORE['amplify_token'],
            DATA_STORE['candidates'],
            DATA_STORE['params']
        )
        updated_candidates, plot_data, scales = result
        DATA_STORE['candidates'] = updated_candidates
        DATA_STORE['optimization_scales'] = scales
        save_settings(DATA_STORE)
        return jsonify({
            "status": "success", 
            "candidates": DATA_STORE['candidates'],
            "plot_data": plot_data,
            "scales": scales
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/bbo_step', methods=['POST'])
def bbo_step():
    global DATA_STORE
    req = request.json
    if req.get('amplify_token'): DATA_STORE['amplify_token'] = req.get('amplify_token')
    if req.get('params'): DATA_STORE['params'] = req.get('params')
    save_settings(DATA_STORE)
    
    for c in DATA_STORE['candidates']:
        if c.get('user_rating', 0) > 0:
            DATA_STORE['bbo_history'].append({
                "attributes": c.get('attributes', {}),
                "type": c.get('type'),
                "relevance": c.get('relevance', 0.5),
                "rating": c['user_rating']
            })
    
    try:
        result = LogicHandler.run_bbo_optimization(
            DATA_STORE['amplify_token'],
            DATA_STORE['candidates'],
            DATA_STORE['bbo_history'],
            DATA_STORE['params']
        )
        updated_candidates, plot_data, scales = result
        DATA_STORE['candidates'] = updated_candidates
        DATA_STORE['optimization_scales'] = scales
        save_settings(DATA_STORE)
        
        return jsonify({
            "status": "success", 
            "candidates": DATA_STORE['candidates'],
            "history_count": len(DATA_STORE['bbo_history']),
            "plot_data": plot_data,
            "scales": scales
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/bbo_reset', methods=['POST'])
def bbo_reset():
    global DATA_STORE
    DATA_STORE['bbo_history'] = []
    for c in DATA_STORE['candidates']: c['user_rating'] = 0
    save_settings(DATA_STORE)
    return jsonify({"status": "success"})

@app.route('/api/generate_draft', methods=['POST'])
def generate_draft():
    global DATA_STORE
    req = request.json
    if req.get('gemini_key'):
        DATA_STORE['gemini_key'] = req.get('gemini_key')
        save_settings(DATA_STORE)

    selected = [c for c in DATA_STORE['candidates'] if c.get('selected')]
    if not selected:
        return jsonify({"status": "error", "message": "No optimized elements selected."}), 400
    try:
        summary, article = LogicHandler.generate_draft(
            DATA_STORE['gemini_key'], selected, DATA_STORE['params']
        )
        DATA_STORE['draft_summary'] = summary
        DATA_STORE['draft_article'] = article
        save_settings(DATA_STORE)
        return jsonify({"status": "success", "summary": summary, "article": article})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/save_draft_edit', methods=['POST'])
def save_draft_edit():
    global DATA_STORE
    DATA_STORE['draft_article'] = request.json.get('article')
    DATA_STORE['additional_instruction'] = request.json.get('instruction')
    save_settings(DATA_STORE)
    return jsonify({"status": "success"})

@app.route('/api/generate_final', methods=['POST'])
def generate_final():
    global DATA_STORE
    req = request.json
    if req and req.get('gemini_key'):
        DATA_STORE['gemini_key'] = req.get('gemini_key')
        save_settings(DATA_STORE)

    try:
        final_text = LogicHandler.generate_final(
            DATA_STORE['gemini_key'], DATA_STORE['draft_article'], DATA_STORE['additional_instruction']
        )
        DATA_STORE['final_text'] = final_text
        save_settings(DATA_STORE)
        return jsonify({"status": "success", "final_text": final_text})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/download')
def download_file():
    global DATA_STORE
    mem = io.BytesIO()
    mem.write(DATA_STORE.get('final_text', '').encode('utf-8'))
    mem.seek(0)
    response = make_response(send_file(mem, as_attachment=True, download_name='novel_scene.txt', mimetype='text/plain'))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

@app.route('/api/download_settings')
def download_settings():
    global DATA_STORE
    mem = io.BytesIO()
    mem.write(json.dumps(DATA_STORE, ensure_ascii=False, indent=2).encode('utf-8'))
    mem.seek(0)
    response = make_response(send_file(mem, as_attachment=True, download_name='settings.json', mimetype='application/json'))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/api/settings/upload', methods=['POST'])
def upload_settings():
    global DATA_STORE
    if 'file' not in request.files: return jsonify({"status": "error"}), 400
    f = request.files['file']
    if f.filename == '': return jsonify({"status": "error", "message": "No file selected"}), 400
    
    try:
        f.seek(0)
        content_bytes = f.read()
        if len(content_bytes) == 0: return jsonify({"status": "error", "message": "File is empty"}), 400
            
        try: content = content_bytes.decode('utf-8-sig').strip()
        except UnicodeDecodeError: content = content_bytes.decode('utf-8', errors='ignore').strip()
        
        if not content: return jsonify({"status": "error", "message": "File content is empty"}), 400
        if content.lstrip().startswith('<'):
             return jsonify({ "status": "error", "message": "HTMLファイルです。正しいJSONファイルを使用してください。" }), 400

        data = json.loads(content)
        DATA_STORE.update(data)
        
        if 'candidates' in DATA_STORE and 'params' in DATA_STORE:
            try:
                metrics = LogicHandler.recalculate_scales(DATA_STORE['candidates'], DATA_STORE['params'])
                if metrics:
                    if 'optimization_scales' not in DATA_STORE or not DATA_STORE['optimization_scales']:
                        DATA_STORE['optimization_scales'] = {}
                    DATA_STORE['optimization_scales'].update(metrics)
            except Exception as e:
                print(f"Warning: Failed to recalculate scales on upload: {e}")

        save_settings(DATA_STORE)
        return jsonify({"status": "success", "settings": DATA_STORE})
    except json.JSONDecodeError as je:
        return jsonify({"status": "error", "message": f"JSON Parse Error: {str(je)}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    # 【修正】Renderの環境変数からポートを取得し、外部(0.0.0.0)に公開する
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)