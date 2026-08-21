"""
Speech-to-text, abstracted behind a single transcribe() function so the
rest of the app never depends on which engine is installed.

Default backend: faster-whisper (CTranslate2-based, runs fully offline
after the model is downloaded once, no cloud calls). Swap by editing this
file only -- e.g. to whisper.cpp via subprocess, or Vosk for a smaller
footprint.
"""
import os
import tempfile

_model = None


def _get_model():
    """Lazily load the whisper model so importing this module doesn't
    require the (heavy) dependency unless transcription is actually used."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        from app.config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
        _model = WhisperModel(
            WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
        )
    return _model


def transcribe(audio_bytes: bytes, filename_hint: str = "note.wav") -> str:
    """
    Transcribe raw audio bytes to text. Writes to a temp file since
    faster-whisper (like most STT libs) expects a file path or file-like
    object with a real audio container, not raw bytes.
    """
    suffix = os.path.splitext(filename_hint)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model = _get_model()
        segments, _info = model.transcribe(tmp_path, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        os.unlink(tmp_path)


class TranscriptionError(Exception):
    pass
