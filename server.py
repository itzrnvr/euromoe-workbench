"""
EuroMoE Interactive Workbench — step-by-step LLM debugger.
Supports: pause/inspect/modify at any point, force experts, force tokens,
disable experts, attention sparsity, per-step timing.
"""
import sys, json, time, gc, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = Path("models/euromoe")
DEVICE = torch.device("cuda")
app = Flask(__name__)

model = None; tokenizer = None
n_layers = 0; n_experts = 0; n_heads = 0; hidden_size = 0

# Session state
session = {
    "input_ids": None,
    "generated_ids": [],
    "chat_history": [],
    "prompt": "",
}

# Modifications (applied to next step)
mods = {
    "forced_experts": {},     # {layer_idx: [expert_ids]}
    "forced_token": None,     # token_id to force
    "disabled_experts": {},   # {layer_idx: [expert_ids]}
    "attn_sparsity": 0.0,     # threshold (0=no sparsity, 0.5=zero weights below 0.5*max)
    "n_active_experts": 8,    # number of experts to route to
}

# Captured data from last step
last_step_data = {}

# Timing
timing = {"attn": [], "moe": [], "total": 0}

# Hooks
_hooks_active = False

def load_model():
    global model, tokenizer, n_layers, n_experts, n_heads, hidden_size
    print("Loading EuroMoE...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(DEVICE).eval()
    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_local_experts
    n_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    print(f"Loaded: {n_layers}L, {n_experts}E, {n_heads}H, hidden={hidden_size}", flush=True)
    
    # Register hooks
    for li, layer in enumerate(model.model.layers):
        # Gate hook — can override routing
        def make_gate_hook(layer_idx):
            def hook(module, inp, out):
                t0 = time.perf_counter()
                # Capture real routing
                if isinstance(out, tuple) and len(out) >= 3:
                    rl = out[0]; tw = out[1]; ti = out[2]
                    if rl.dim() == 3: rl = rl[:, -1, :]
                    probs = F.softmax(rl.float(), dim=-1).squeeze(0)
                    
                    # Apply modifications — WITH SAFETY CHECKS
                    forced = mods["forced_experts"].get(str(layer_idx)) or mods["forced_experts"].get(layer_idx)
                    disabled = mods["disabled_experts"].get(str(layer_idx)) or mods["disabled_experts"].get(layer_idx)
                    
                    if forced:
                        # SAFETY: validate indices are in range and unique
                        forced = [int(x) for x in forced if isinstance(x, (int, float)) and 0 <= int(x) < n_experts]
                        forced = list(set(forced))  # remove duplicates
                        if len(forced) == 0:
                            forced = list(range(8))  # fallback to default
                        # Clamp count to avoid topk errors
                        forced = forced[:min(len(forced), n_experts)]
                        
                        forced_idx = torch.tensor(forced, device=DEVICE, dtype=torch.long)
                        forced_w = probs[forced_idx]
                        # SAFETY: prevent division by zero
                        w_sum = forced_w.sum().clamp_min(1e-8)
                        forced_w = forced_w / w_sum
                        # SAFETY: check for NaN/Inf
                        if torch.isnan(forced_w).any() or torch.isinf(forced_w).any():
                            forced_w = torch.ones_like(forced_w) / len(forced)  # uniform fallback
                        tw_new = forced_w.unsqueeze(0)
                        ti_new = forced_idx.unsqueeze(0)
                        return (rl, tw_new, ti_new)
                    
                    if disabled:
                        # SAFETY: validate indices
                        disabled = [int(x) for x in disabled if isinstance(x, (int, float)) and 0 <= int(x) < n_experts]
                        probs_modified = probs.clone()
                        for d in disabled:
                            probs_modified[d] = 0
                        # SAFETY: check that at least n_active experts remain
                        n_remaining = (probs_modified > 0).sum().item()
                        n_keep = min(mods["n_active_experts"], n_remaining)
                        if n_keep < 1:
                            # All experts disabled — skip modification, use original
                            pass
                        else:
                            p_sum = probs_modified.sum().clamp_min(1e-8)
                            probs_modified = probs_modified / p_sum
                            # SAFETY: check for NaN
                            if torch.isnan(probs_modified).any() or torch.isinf(probs_modified).any():
                                probs_modified = probs  # fallback to original
                            else:
                                tw_new, ti_new = torch.topk(probs_modified, k=n_keep)
                                tw_new = tw_new / tw_new.sum().clamp_min(1e-8)
                                tw_new = tw_new.unsqueeze(0) if tw_new.dim()==1 else tw_new
                                ti_new = ti_new.unsqueeze(0) if ti_new.dim()==1 else ti_new
                                return (rl, tw_new, ti_new)
                    
                    # Store real routing
                    if _hooks_active:
                        last_step_data.setdefault("routing", []).append({
                            "layer": layer_idx,
                            "indices": ti[-1].cpu().numpy().tolist() if ti.dim()>=1 else ti.cpu().numpy().tolist(),
                            "weights": tw[-1].cpu().numpy().tolist() if tw.dim()>=1 else tw.cpu().numpy().tolist(),
                            "all_probs": probs.cpu().numpy().tolist(),
                        })
                
                gate_time = (time.perf_counter() - t0) * 1000
                return None
            return hook
        layer.mlp.gate.register_forward_hook(make_gate_hook(li))
        
        # Attention timing hook
        def make_attn_timing_hook(layer_idx):
            t_ref = [0.0]
            def pre(module, inp): t_ref[0] = time.perf_counter()
            def post(module, inp, out):
                if _hooks_active:
                    last_step_data.setdefault("attn_ms", []).append({
                        "layer": layer_idx,
                        "ms": (time.perf_counter() - t_ref[0]) * 1000
                    })
            layer.self_attn.register_forward_pre_hook(pre)
            layer.self_attn.register_forward_hook(post)
        make_attn_timing_hook(li)
        
        # MLP timing hook
        def make_mlp_timing_hook(layer_idx):
            t_ref = [0.0]
            def pre(module, inp): t_ref[0] = time.perf_counter()
            def post(module, inp, out):
                if _hooks_active:
                    last_step_data.setdefault("moe_ms", []).append({
                        "layer": layer_idx,
                        "ms": (time.perf_counter() - t_ref[0]) * 1000
                    })
            layer.mlp.register_forward_pre_hook(pre)
            layer.mlp.register_forward_hook(post)
        make_mlp_timing_hook(li)
        
        # Hidden state hook
        def make_hs_hook(layer_idx):
            def hook(module, inp, out):
                if not _hooks_active: return
                hs = out[0] if isinstance(out, tuple) else out
                if hs.dim() == 3:
                    last_vec = hs[:, -1, :].squeeze(0).float().cpu()
                    in_vec = None
                    if isinstance(inp, tuple) and inp[0].dim() == 3:
                        in_vec = inp[0][:, -1, :].squeeze(0).float().cpu()
                    delta = 0
                    if in_vec is not None and in_vec.shape == last_vec.shape:
                        delta = float((last_vec - in_vec).norm() / in_vec.norm().clamp_min(1e-8))
                    last_step_data.setdefault("hidden", []).append({
                        "layer": layer_idx, "norm": float(last_vec.norm()),
                        "delta": delta, "mean": float(last_vec.mean()), "std": float(last_vec.std()),
                    })
            return hook
        layer.register_forward_hook(make_hs_hook(li))

    # Embedding hook
    def embed_hook(module, inp, out):
        if _hooks_active and out.dim() == 3:
            last_step_data["embedding"] = out[:, -1, :].squeeze(0).float().cpu().abs().view(64, -1).mean(dim=1).tolist()
    model.model.embed_tokens.register_forward_hook(embed_hook)


def generate_step():
    """Generate exactly ONE token. WITH GPU SAFETY GUARDS."""
    global _hooks_active, is_generating
    
    # GUARD 1: prevent concurrent generation
    if is_generating:
        return {"error": "Generation in progress, wait..."}
    is_generating = True
    
    # GUARD 2: check GPU memory
    if torch.cuda.is_available():
        free_mb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1048576
        if free_mb < 400:
            torch.cuda.empty_cache()
            free_mb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1048576
            if free_mb < 200:
                is_generating = False
                return {"error": "Low GPU memory (" + str(int(free_mb)) + "MB). Clear conversation."}
    
    # GUARD 3: check sequence length
    if session["input_ids"] is not None and session["input_ids"].shape[1] > 2048:
        is_generating = False
        return {"error": "Context too long (max 2048). Clear conversation."}
    
    # GUARD 4: validate forced token
    if mods["forced_token"] is not None and (not isinstance(mods["forced_token"], int) or mods["forced_token"] < 0):
        mods["forced_token"] = None
    
    last_step_data.clear()
    _hooks_active = True
    t_start = time.perf_counter()
    
    try:
        with torch.inference_mode():
            outputs = model(session["input_ids"], use_cache=True, output_attentions=True)
            logits = outputs.logits[:, -1, :]
            
            # GUARD 5: check for NaN in output (corrupted routing can cause this)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                raise RuntimeError("NaN/Inf detected in model output — resetting modifications")
            
            # Token selection
            if mods["forced_token"] is not None:
                next_id = mods["forced_token"]
                forced = True
            else:
                next_id = logits.argmax(dim=-1).item()
                forced = False
            
            # Top predictions
            top5_vals, top5_idx = torch.topk(logits.float(), 10)
            top5_probs = F.softmax(logits.float(), dim=-1)[0, top5_idx[0]]
            preds = [
                {"t": tokenizer.decode([i.item()]), "id": i.item(), "p": float(top5_probs[j].item())}
                for j, i in enumerate(top5_idx[0])
            ]
            
            # KV cache info
            kv_info = {"n_tokens": 0, "size_mb": 0}
            if outputs.past_key_values:
                total_el = 0
                for lk in outputs.past_key_values:
                    for t in lk:
                        if t is not None and hasattr(t, 'numel'):
                            total_el += t.numel()
                kv_info = {"n_tokens": session["input_ids"].shape[1], "size_mb": round(total_el * 2 / 1048576, 1)}
            
            # Attention weights (last layer, last query)
            attn_data = []
            seq_strs = [tokenizer.decode([t]) for t in session["input_ids"][0]]
            if outputs.attentions:
                for li, attn in enumerate(outputs.attentions):
                    w = attn[0, :, -1, :].float().mean(dim=0).cpu()
                    attn_data.append({"layer": li, "weights": [round(v, 4) for v in w.tolist()]})
            
            total_time = (time.perf_counter() - t_start) * 1000
            attn_total = sum(a["ms"] for a in last_step_data.get("attn_ms", []))
            moe_total = sum(a["ms"] for a in last_step_data.get("moe_ms", []))
            
            token_str = tokenizer.decode([next_id])
            
            result = {
                "token": token_str, "token_id": next_id, "forced": forced,
                "text": tokenizer.decode(session["generated_ids"] + [next_id]),
                "time_ms": round(total_time, 1), "attn_total": round(attn_total, 1),
                "moe_total": round(moe_total, 1),
                "other_ms": round(total_time - attn_total - moe_total, 1),
                "preds": preds, "kv_cache": kv_info,
                "hidden": last_step_data.get("hidden", []),
                "routing": last_step_data.get("routing", []),
                "attn_ms": last_step_data.get("attn_ms", []),
                "moe_ms": last_step_data.get("moe_ms", []),
                "embedding": last_step_data.get("embedding", []),
                "attentions": attn_data, "seq_strs": seq_strs,
                "mods": {k: v for k, v in mods.items()},
            }
            
            # Update session
            session["generated_ids"].append(next_id)
            session["input_ids"] = torch.cat([session["input_ids"], torch.tensor([[next_id]], device=DEVICE)], dim=-1)
            mods["forced_token"] = None
            
            del outputs, logits
            torch.cuda.empty_cache()
            return result
    
    except RuntimeError as e:
        # GPU error — reset everything to safe state
        print(f"GPU SAFETY: {e}", flush=True)
        torch.cuda.empty_cache()
        # Reset all modifications that might have caused the issue
        mods["forced_experts"] = {}
        mods["disabled_experts"] = {}
        mods["forced_token"] = None
        mods["n_active_experts"] = 8
        return {"error": f"GPU safety stop: {e}. All modifications reset. Try again."}
    
    finally:
        # ALWAYS clean up, even on error
        _hooks_active = False
        is_generating = False
        torch.cuda.empty_cache()


# ===== ROUTES =====
@app.route("/")
def index():
    with open(str(Path(__file__).parent / "workbench.html"), "r", encoding="utf-8") as f:
        content = f.read()
    from flask import Response
    return Response(content, mimetype="text/html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
@app.route("/api/status")
def api_status():
    return jsonify({
        "loaded": model is not None,
        "n_layers": n_layers, "n_experts": n_experts,
        "n_heads": n_heads, "hidden_size": hidden_size,
        "chat_history": session["chat_history"],
        "kv_cache": {"n_tokens": session["input_ids"].shape[1] if session["input_ids"] is not None else 0},
        "gpu_memory": {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 1),
            "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1048576, 1),
        } if torch.cuda.is_available() else {},
        "mods": mods,
        "n_generated": len(session["generated_ids"]),
    })

@app.route("/api/start", methods=["POST"])
def api_start():
    """Start a new generation session with a prompt."""
    data = request.json
    prompt = data.get("prompt", "")
    
    full_prompt = ""
    for msg in session["chat_history"]:
        full_prompt += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n"
    full_prompt += f"User: {prompt}\nAssistant:"
    
    session["input_ids"] = tokenizer(full_prompt, return_tensors="pt").input_ids.to(DEVICE)
    session["generated_ids"] = []
    session["prompt"] = prompt
    session["chat_history"].append({"user": prompt, "assistant": ""})
    
    return jsonify({"status": "started", "prompt_len": session["input_ids"].shape[1]})

@app.route("/api/step", methods=["POST"])
def api_step():
    """Generate one token and return full state."""
    if session["input_ids"] is None:
        return jsonify({"error": "No active session. Call /api/start first."}), 400
    
    # Check for EOS
    if session["generated_ids"] and session["generated_ids"][-1] == tokenizer.eos_token_id:
        return jsonify({"done": True})
    
    result = generate_step()
    
    # Check EOS
    if result["token_id"] == tokenizer.eos_token_id:
        result["done"] = True
        session["chat_history"][-1]["assistant"] = tokenizer.decode(session["generated_ids"])
    
    return jsonify(result)

@app.route("/api/mods", methods=["POST"])
def api_set_mods():
    """Set modifications for next step."""
    data = request.json
    for key in ["forced_experts", "forced_token", "disabled_experts", "attn_sparsity", "n_active_experts"]:
        if key in data:
            mods[key] = data[key]
    return jsonify({"status": "set", "mods": mods})

@app.route("/api/reset_mods", methods=["POST"])
def api_reset_mods():
    """Reset all modifications."""
    mods["forced_experts"] = {}
    mods["forced_token"] = None
    mods["disabled_experts"] = {}
    mods["attn_sparsity"] = 0.0
    mods["n_active_experts"] = 8
    return jsonify({"status": "reset"})

@app.route("/api/clear", methods=["POST"])
def api_clear():
    """Clear session."""
    session["input_ids"] = None
    session["generated_ids"] = []
    session["chat_history"] = []
    gc.collect()
    torch.cuda.empty_cache()
    return jsonify({"status": "cleared"})

@app.route("/api/tokenize", methods=["POST"])
def api_tokenize():
    data = request.json
    ids = tokenizer(data["prompt"], return_tensors="pt").input_ids[0].tolist()
    return jsonify({"tokens": [{"id": i, "text": tokenizer.decode([i])} for i in ids]})

@app.route("/api/force_token", methods=["POST"])
def api_force_token():
    """Force the next generated token."""
    data = request.json
    token_str = data.get("token", "")
    ids = tokenizer(token_str, return_tensors="pt").input_ids[0].tolist()
    if ids:
        mods["forced_token"] = ids[0]
        return jsonify({"status": "forced", "token_id": ids[0], "token": tokenizer.decode([ids[0]])})
    return jsonify({"error": "Could not tokenize"}), 400

@app.route("/api/gpu_info")
def api_gpu():
    if not torch.cuda.is_available():
        return jsonify({})
    return jsonify({
        "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 1),
        "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1048576, 1),
        "free_mb": round((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1048576, 1),
    })

if __name__ == "__main__":
    load_model()
    print("\nDashboard at http://localhost:5000", flush=True)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
