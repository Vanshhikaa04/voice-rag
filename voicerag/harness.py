"""The harness: runs every stage with timing, retries, structured I/O and error recovery.

ask() NEVER raises. Every outcome - answer, refusal, degraded fallback, crash - comes back as a
RagResponse with a status, per-stage timings and a trace id, so callers (and the benchmark) can
treat it as plain data.
"""
import json
import logging
import random
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import numpy as np

from . import guardrails as gr
from .config import Config
from .generator import Answer, ExtractiveGenerator
from .guardrails import Status

log = logging.getLogger("voicerag")

REFUSALS = {
    Status.NO_QUERY: "Sorry, I didn't catch a question there. Could you say it again?",
    Status.UNSAFE: "I can't help with that request.",
    Status.LOW_CONFIDENCE: "I couldn't find that in my documents, so I won't guess.",
    Status.UNGROUNDED: "I couldn't produce an answer I can back up with my documents.",
    Status.ERROR: "Something went wrong on my side. Please try again.",
}


@dataclass
class RagResponse:
    status: Status
    answer: str
    question: str = ""
    sources: list[dict] = field(default_factory=list)
    confidence: dict = field(default_factory=dict)       # dense / coverage / grounding / stt
    generator: str = ""
    degraded: bool = False                               # True if a fallback path was used
    reason: str = ""
    timings_ms: dict = field(default_factory=dict)
    total_ms: float = 0.0
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=lambda o: getattr(o, "value", str(o)), indent=2)


def with_retry(fn, attempts: int, base_delay: float, what: str):
    """Exponential backoff with jitter. Re-raises the last error once attempts are exhausted."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise
            delay = base_delay * (2 ** i) * (1 + random.random() * 0.25)
            log.warning("%s failed (%s: %s); retry %d/%d in %.0f ms", what, type(e).__name__, e, i + 1,
                        attempts - 1, delay * 1000)
            time.sleep(delay)


class VoiceRAG:
    def __init__(self, cfg: Config, index, transcriber=None, generator=None):
        self.cfg = cfg
        self.index = index
        self.transcriber = transcriber
        self.fallback = ExtractiveGenerator(index)
        self.generator = generator or self.fallback

    @contextmanager
    def _stage(self, timings: dict, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            timings[name] = (time.perf_counter() - t0) * 1000

    def ask(self, text: str | None = None, audio: np.ndarray | None = None) -> RagResponse:
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        resp = RagResponse(Status.ERROR, REFUSALS[Status.ERROR])
        try:
            resp = self._run(text, audio, timings)
        except Exception as e:                                       # last-resort recovery
            log.exception("pipeline crashed")
            resp.reason = f"{type(e).__name__}: {e}"
        resp.timings_ms = {k: round(v, 3) for k, v in timings.items()}
        resp.total_ms = round((time.perf_counter() - t_start) * 1000, 3)
        log.info(json.dumps({"trace": resp.trace_id, "status": resp.status.value, "ms": resp.total_ms,
                             "degraded": resp.degraded, "reason": resp.reason}))
        return resp

    # ------------------------------------------------------------------------------------------
    def _refuse(self, status: Status, question: str, reason: str, conf: dict, hits=None) -> RagResponse:
        msg = REFUSALS[status]
        if status == Status.LOW_CONFIDENCE and hits:
            msg += f" The closest topic I have is \"{hits[0].chunk.title}\"."
        return RagResponse(status, msg, question, confidence=conf, reason=reason)

    def _run(self, text, audio, timings) -> RagResponse:
        cfg, conf = self.cfg, {}

        # 1. speech -> text -------------------------------------------------------------------
        stt_conf = None
        if audio is not None:
            if self.transcriber is None:
                raise RuntimeError("audio supplied but no transcriber configured")
            with self._stage(timings, "stt"):
                tr = with_retry(lambda: self.transcriber.transcribe(audio), 2, cfg.retry_base_delay, "stt")
            text, stt_conf = tr.text, tr.confidence
            conf["stt"] = round(stt_conf, 3)
        question = (text or "").strip()

        # 2. input guardrail ------------------------------------------------------------------
        with self._stage(timings, "guard_in"):
            v = gr.check_input(question, stt_conf, cfg)
        if not v.ok:
            return self._refuse(v.status, question, v.reason, conf)

        # 3. retrieval + retrieval guardrail ---------------------------------------------------
        with self._stage(timings, "retrieve"):
            hits = gr.filter_hits(self.index.search(question, cfg.top_k))
        if hits:
            conf["dense"], conf["coverage"] = round(hits[0].dense, 3), round(max(h.coverage for h in hits), 3)
        with self._stage(timings, "guard_retrieval"):
            v = gr.check_retrieval(hits, cfg)
        if not v.ok:
            return self._refuse(v.status, question, v.reason, conf, hits)

        # 4. generation, with retry then fallback to the local extractive generator -------------
        degraded, reason = False, ""
        with self._stage(timings, "generate"):
            try:
                ans = with_retry(lambda: self.generator.generate(question, hits),
                                 cfg.retry_attempts, cfg.retry_base_delay, "generate")
            except Exception as e:
                log.error("primary generator failed, using extractive fallback: %s", e)
                ans, degraded, reason = self.fallback.generate(question, hits), True, f"fallback: {e}"

        # 5. output guardrail (grounding); regenerate extractively if the LLM drifted ----------
        with self._stage(timings, "guard_out"):
            if ans.text.strip().lower().startswith("i don't know"):
                ans = Answer("", [], ans.generator)
            v = gr.check_output(ans.text, hits, cfg) if ans.text else gr.Verdict(False, Status.LOW_CONFIDENCE, "generator abstained")
            if not v.ok and ans.generator != self.fallback.name:
                ans, degraded, reason = self.fallback.generate(question, hits), True, f"fallback: {v.reason}"
                v = gr.check_output(ans.text, hits, cfg) if ans.text else gr.Verdict(False, Status.LOW_CONFIDENCE, "no supporting sentence")
            if ans.text:
                conf["grounding"] = round(gr.grounding_score(ans.text, hits), 3)
        if not v.ok:
            return self._refuse(v.status, question, v.reason, conf, hits)

        return RagResponse(Status.ANSWERED, ans.text, question, ans.sources, conf, ans.generator, degraded, reason)
