#!/usr/bin/env python3
"""dllama-proxy SYNC: bridge Hermes ↔ dllama-api.
Force stream=false hacia dllama-api y reformat respuesta. SSE si Hermes pide stream."""
import json
import logging
import time
import uuid
import requests
from flask import Flask, Response, request, jsonify

DLLAMA_URL = "http://127.0.0.1:9999/v1"
MODEL_NAME = "qwen3"
TIMEOUT = 600

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dllama-proxy")
app = Flask(__name__)


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "dllama-cluster"}],
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def _normalize_completion(j):
    j.setdefault("id", f"chatcmpl-{uuid.uuid4().hex[:12]}")
    j.setdefault("object", "chat.completion")
    j.setdefault("created", int(time.time()))
    j.setdefault("model", MODEL_NAME)
    new_choices = []
    for c in j.get("choices", []):
        if not c.get("finish_reason"):
            c["finish_reason"] = "stop"
        if "index" not in c:
            c["index"] = 0
        new_choices.append(c)
    j["choices"] = new_choices
    j.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return j


def _sse_chunks(completion):
    cid = completion["id"]
    created = completion["created"]
    model = completion["model"]
    yield "data: " + json.dumps({
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }) + "\n\n"
    for choice in completion["choices"]:
        content = (choice.get("message") or {}).get("content", "")
        if content:
            yield "data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }) + "\n\n"
        yield "data: " + json.dumps({
            "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": choice.get("index", 0), "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}],
        }) + "\n\n"
    yield "data: [DONE]\n\n"


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    body = request.get_json(force=True, silent=True) or {}
    wants_stream = bool(body.get("stream", False))
    body["stream"] = False

    log.info(f"→ chat msgs={len(body.get('messages', []))} stream={wants_stream} max_tokens={body.get('max_tokens')}")

    try:
        r = requests.post(
            f"{DLLAMA_URL}/chat/completions",
            json=body,
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json", "Connection": "close"},
        )
        completion = r.json()
    except Exception as e:
        log.error(f"dllama-api error: {e}")
        return jsonify({"error": {"message": str(e), "type": "upstream_error"}}), 502

    completion = _normalize_completion(completion)
    log.info(f"← {completion['usage']}")

    if wants_stream:
        return Response(_sse_chunks(completion), mimetype="text/event-stream")
    return jsonify(completion)


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="127.0.0.1", port=8000, threads=8)
