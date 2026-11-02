"""Speech gateway. Sarvam suits Hindi and Hinglish; ElevenLabs, OpenAI and local Whisper are alternatives."""
import logging
import tempfile
from functools import lru_cache
from pathlib import Path

import httpx

from . import llm
from .config import keys, settings

log = logging.getLogger("nexus.voice")
SARVAM = "https://api.sarvam.ai"
ELEVEN = "https://api.elevenlabs.io/v1"
_TIMEOUT = 120


class VoiceUnavailable(RuntimeError):
    pass


def stt_provider() -> str:
    k, s = keys(), settings()
    if s.voice_provider in ("sarvam", "openai", "local"):
        return s.voice_provider
    return "sarvam" if k.sarvam_api_key else "openai" if k.openai_api_key else "local"


def tts_provider() -> str | None:
    k, s = keys(), settings()
    if s.voice_provider in ("sarvam", "elevenlabs", "openai"):
        return s.voice_provider
    return "sarvam" if k.sarvam_api_key else "elevenlabs" if k.elevenlabs_api_key else "openai" if k.openai_api_key else None


@lru_cache
def _whisper():
    from faster_whisper import WhisperModel

    return WhisperModel(settings().whisper_model, device="cpu", compute_type="int8")


def _local(data: bytes, filename: str, language: str | None) -> dict:
    # PyAV inside faster-whisper decodes the audio track of any common audio or video file
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / f"input{Path(filename).suffix or '.bin'}"
        path.write_bytes(data)
        segments, info = _whisper().transcribe(str(path), language=language, vad_filter=True, beam_size=1)
        text = " ".join(s.text.strip() for s in segments).strip()
    return {"text": text, "language": info.language, "provider": "local"}


def transcribe(data: bytes, filename: str, language: str | None, workspace_id: str | None) -> dict:
    """language is a hint ("hi", "en"); None lets the provider detect it."""
    provider = stt_provider()
    if provider == "local":
        return _local(data, filename, language)
    files = {"file": (filename, data)}
    if provider == "sarvam":
        form = {"model": "saarika:v2.5", "language_code": {"hi": "hi-IN", "en": "en-IN"}.get(language or "", "unknown")}
        r = httpx.post(f"{SARVAM}/speech-to-text", headers={"api-subscription-key": keys().sarvam_api_key},
                       data=form, files=files, timeout=_TIMEOUT)
        r.raise_for_status()
        d = r.json()
        return {"text": d.get("transcript", ""), "language": (d.get("language_code") or "")[:2], "provider": "sarvam"}
    form = {"model": "whisper-1", "response_format": "verbose_json"}
    if language:
        form["language"] = language
    k = keys()
    r = httpx.post(f"{k.openai_base_url.rstrip('/')}/audio/transcriptions",
                   headers={"Authorization": f"Bearer {k.openai_api_key}"}, data=form, files=files, timeout=_TIMEOUT)
    r.raise_for_status()
    d = r.json()
    llm.record(workspace_id, "stt", "openai", "whisper-1", 0, 0)
    code = {"hindi": "hi", "english": "en"}.get(d.get("language", ""), d.get("language", ""))
    return {"text": d.get("text", "").strip(), "language": code, "provider": "openai"}


def speak(text: str, language: str = "en") -> tuple[bytes, str]:
    """Returns (audio, mime). language is "en", "hi" or "hinglish"."""
    provider, k = tts_provider(), keys()
    text = text[:2400]
    if provider == "sarvam":
        import base64

        r = httpx.post(f"{SARVAM}/text-to-speech", headers={"api-subscription-key": k.sarvam_api_key},
                       json={"text": text[:1500], "target_language_code": "en-IN" if language == "en" else "hi-IN",
                             "model": "bulbul:v2", "speaker": "anushka"}, timeout=_TIMEOUT)
        r.raise_for_status()
        return base64.b64decode(r.json()["audios"][0]), "audio/wav"
    if provider == "elevenlabs":
        if not k.elevenlabs_voice_id:
            raise VoiceUnavailable("Set ELEVENLABS_VOICE_ID")
        r = httpx.post(f"{ELEVEN}/text-to-speech/{k.elevenlabs_voice_id}", headers={"xi-api-key": k.elevenlabs_api_key},
                       json={"text": text, "model_id": "eleven_multilingual_v2"}, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.content, "audio/mpeg"
    if provider == "openai":
        r = httpx.post(f"{k.openai_base_url.rstrip('/')}/audio/speech", headers={"Authorization": f"Bearer {k.openai_api_key}"},
                       json={"model": "tts-1", "voice": "alloy", "input": text}, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.content, "audio/mpeg"
    raise VoiceUnavailable("No text-to-speech provider configured (set SARVAM_API_KEY, ELEVENLABS_API_KEY or OPENAI_API_KEY)")
