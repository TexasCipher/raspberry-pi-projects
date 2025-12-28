from flask import Flask, request, jsonify, render_template, send_file
import os
import json
from threading import Lock

import generate

app = Flask(__name__)

# Simple in-memory state with file-backed persistence
STATE_FILE = "chat_state.json"
state_lock = Lock()
if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        state = {"history": []}
else:
    state = {"history": []}


def save_state():
    with state_lock:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user = data.get("message", "")
    model = data.get("model", "distilgpt2")
    device = data.get("device", "cpu")
    dry = data.get("dry_run", True)

    if not user:
        return jsonify({"error": "no message"}), 400

    # Append to history
    with state_lock:
        state.setdefault("history", []).append({"role": "user", "text": user})

    # Build pipeline (reuse simple configuration)
    p, dev = generate.init_pipeline(model=model, device_opt=device, dry_run=dry)

    # Build prompt by including last few turns
    history = state.get("history", [])
    last_turns = []
    # include up to last 6 messages
    for msg in history[-6:]:
        prefix = "User: " if msg["role"] == "user" else "AI: "
        last_turns.append(prefix + msg["text"])

    prompt = "\n".join(last_turns) + "\nAI:"

    out = generate.generate_text(
        p,
        prompt,
        max_new_tokens=int(data.get("max_new_tokens", 120)),
        do_sample=not data.get("no_sample", False),
        temperature=float(data.get("temperature", 0.8)),
        top_k=int(data.get("top_k", 0)),
        top_p=float(data.get("top_p", 0.92)),
        repetition_penalty=float(data.get("repetition_penalty", 1.15)),
        num_return_sequences=int(data.get("num_return_sequences", 1)),
    )

    try:
        text = out[0].get("generated_text") if isinstance(out[0], dict) else str(out[0])
    except Exception:
        text = str(out)

    # Save assistant reply
    with state_lock:
        state.setdefault("history", []).append({"role": "assistant", "text": text.strip()})
    save_state()

    return jsonify({"reply": text.strip()})


if __name__ == "__main__":
    # For local dev only
    app.run(host="0.0.0.0", port=5000, debug=True)
