"""Serves the web UI and exposes the voicerag pipeline as a RAG endpoint on one port.

GET  /          -> web/index.html (and other files in web/)
POST /query     -> {"question", "top_k"?, "chunks"?: [{"id","source","text"}]}
                   Uses the uploaded chunks if given, otherwise the documents in data/docs.
                   Replies {"answer", "citations", "status", "confidence", "timings_ms", "total_ms"}.
"""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from voicerag.chunking import Chunk
from voicerag.config import Config
from voicerag.harness import VoiceRAG
from voicerag.index import Index

ROOT = Path(__file__).parent
CFG = Config()


def make_generator():
    """OpenAI if OPENAI_API_KEY is set, else Claude if ANTHROPIC_API_KEY is set, else None (extractive)."""
    try:
        if os.environ.get("OPENAI_API_KEY"):
            from voicerag.generator import OpenAIGenerator
            return OpenAIGenerator()
        if os.environ.get("ANTHROPIC_API_KEY"):
            from voicerag.generator import ClaudeGenerator
            return ClaudeGenerator()
    except Exception as e:
        print(f"LLM unavailable ({type(e).__name__}: {e}); using extractive answers")
    return None


GEN = make_generator()
DEFAULT = VoiceRAG(CFG, Index.from_dir(CFG), generator=GEN)
_uploaded: dict[str, VoiceRAG] = {}


def rag_for(chunks: list[dict]) -> VoiceRAG:
    key = hashlib.sha1(json.dumps(chunks, sort_keys=True).encode()).hexdigest()
    if key not in _uploaded:
        if len(_uploaded) >= 8:
            _uploaded.pop(next(iter(_uploaded)))
        idx = Index(CFG)
        idx.build([Chunk(c["id"], c["source"], Path(c["source"]).stem, c["text"], "uploaded") for c in chunks])
        _uploaded[key] = VoiceRAG(CFG, idx, generator=GEN)
    return _uploaded[key]


class Handler(SimpleHTTPRequestHandler):
    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path.split("?")[0] == "/query":
            return self._send(200, {"info": "POST JSON {\"question\": \"...\"} to this URL"})
        super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] != "/query":
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            chunks = body.get("chunks") or []
            rag = rag_for(chunks) if chunks else DEFAULT
            if body.get("top_k"):                                    # Config is frozen: use a per-request copy
                rag = copy.copy(rag)
                rag.cfg = dataclasses.replace(rag.cfg, top_k=max(1, min(int(body["top_k"]), 20)))
            r = rag.ask(str(body.get("question", "")))
        except Exception as e:
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})
        self._send(200, {
            "answer": r.answer, "status": r.status.value,
            "citations": [f"[{s['n']}] {s['chunk_id']}" for s in r.sources],
            "confidence": r.confidence, "timings_ms": r.timings_ms, "total_ms": r.total_ms,
        })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    handler = partial(Handler, directory=str(ROOT / "web"))
    print(f"generator: {GEN.name + ' (' + GEN.model + ')' if GEN else 'extractive (no API key)'}")
    print(f"voicerag server on http://localhost:{a.port}  (UI at /, RAG endpoint POST /query)")
    ThreadingHTTPServer(("0.0.0.0", a.port), handler).serve_forever()
