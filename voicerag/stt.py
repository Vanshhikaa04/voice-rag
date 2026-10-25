"""Speech input: mic capture with silence detection, WAV loading, and a local Whisper transcriber."""
import math
import wave
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000


@dataclass
class Transcript:
    text: str
    confidence: float          # 0..1, derived from Whisper's mean token log-prob


class WhisperTranscriber:
    """faster-whisper (CTranslate2) running locally. `tiny.en` + int8 is the speed setting."""

    def __init__(self, model: str = "tiny.en", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device="cpu", compute_type=compute_type)
        self.model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))     # warm-up

    def transcribe(self, audio: np.ndarray) -> Transcript:
        segments, _ = self.model.transcribe(
            audio, beam_size=1, language="en", vad_filter=True, condition_on_previous_text=False,
        )
        segs = list(segments)
        if not segs:
            return Transcript("", 0.0)
        text = " ".join(s.text.strip() for s in segs).strip()
        conf = math.exp(sum(s.avg_logprob for s in segs) / len(segs))
        return Transcript(text, float(min(conf, 1.0)))


def load_wav(path: str) -> np.ndarray:
    """16-bit PCM WAV -> mono float32 at 16 kHz (linear resampling if needed)."""
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, "only 16-bit PCM WAV is supported"
        rate, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if rate != SAMPLE_RATE:
        n = int(len(data) * SAMPLE_RATE / rate)
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)
    return data


def record_until_silence(max_seconds: float = 10.0, silence_seconds: float = 0.7,
                         threshold: float = 0.01) -> np.ndarray:
    """Record from the default mic; stop after `silence_seconds` of quiet following speech."""
    import sounddevice as sd

    block = int(0.05 * SAMPLE_RATE)
    frames, heard, quiet = [], False, 0.0
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block) as stream:
        while len(frames) * 0.05 < max_seconds:
            data, _ = stream.read(block)
            frames.append(data[:, 0].copy())
            loud = float(np.sqrt(np.mean(data ** 2))) > threshold
            heard = heard or loud
            quiet = 0.0 if loud else quiet + 0.05
            if heard and quiet >= silence_seconds:
                break
    return np.concatenate(frames)
