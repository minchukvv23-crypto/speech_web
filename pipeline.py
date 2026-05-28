import logging
import os
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, load_silero_vad

from app.audio_enhance import enhance_audio
from app.diarization_gpu import run_diarization_gpu
from app.utils.cleanup import cleanup_paths

log = logging.getLogger(__name__)

_PROGRESS_CB = None
_VAD_MODEL = None
_WHISPER_MODELS = {}


def set_progress_callback(callback: Optional[Callable[[str, int, Optional[int], str], None]]):
    global _PROGRESS_CB
    _PROGRESS_CB = callback


def update_progress(stage: str, progress: int, eta_sec: Optional[int] = None, status: str = "processing"):
    if _PROGRESS_CB:
        _PROGRESS_CB(stage, progress, eta_sec, status)


def _get_vad_model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


def _get_whisper_model(model_size: str = "large", device: str = "cuda"):
    global _WHISPER_MODELS

    key = (model_size, device)
    if key not in _WHISPER_MODELS:
        compute_type = "float16" if device == "cuda" else "int8"
        _WHISPER_MODELS[key] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
    return _WHISPER_MODELS[key]


def ensure_mono_16k(input_path: str, output_path: str):
    import ffmpeg

    (
        ffmpeg
        .input(input_path)
        .output(
            output_path,
            acodec="pcm_s16le",
            ac=1,
            ar=16000,
            format="wav",
        )
        .overwrite_output()
        .run(quiet=True)
    )


def load_wav_16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != 16000:
        raise ValueError(f"Expected 16kHz wav, got {sr}")

    return audio


def run_vad(audio_16k: np.ndarray) -> List[Dict]:
    model = _get_vad_model()

    wav_tensor = torch.from_numpy(audio_16k)
    speech_timestamps = get_speech_timestamps(
        wav_tensor,
        model,
        sampling_rate=16000,
    )

    results = []
    for seg in speech_timestamps:
        results.append(
            {
                "start": float(seg["start"]) / 16000.0,
                "end": float(seg["end"]) / 16000.0,
            }
        )

    return results


def speaker_at_time(diar_segments: List[Dict], t: float) -> str:
    for seg in diar_segments:
        if seg["start"] <= t <= seg["end"]:
            return seg["speaker"]
    return "UNK"


def run_asr_over_vad(
    audio_16k: np.ndarray,
    vad_segments: List[Dict],
    device: str,
    model_size: str,
) -> List[Dict]:
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available. Falling back to CPU for ASR.")
        device = "cpu"

    whisper = _get_whisper_model(model_size=model_size, device=device)

    results = []
    sr = 16000

    for i, seg in enumerate(vad_segments, start=1):
        s = float(seg["start"])
        e = float(seg["end"])
        s_i = int(s * sr)
        e_i = int(e * sr)
        chunk = audio_16k[s_i:e_i]

        if len(chunk) < int(0.1 * sr):
            continue

        segments, _ = whisper.transcribe(
            chunk,
            language=None,
            beam_size=5,
        )

        for out_seg in segments:
            abs_start = float(out_seg.start) + s
            abs_end = float(out_seg.end) + s
            text = (out_seg.text or "").strip()

            if text:
                results.append(
                    {
                        "start": abs_start,
                        "end": abs_end,
                        "text": text,
                    }
                )

        log.debug(
            "ASR chunk %d/%d processed: %.2f-%.2f sec",
            i, len(vad_segments), s, e
        )

    return results


def attach_speakers(asr_segments: List[Dict], diar_segments: List[Dict]) -> List[Dict]:
    merged = []

    for seg in asr_segments:
        mid = (seg["start"] + seg["end"]) / 2.0
        spk = speaker_at_time(diar_segments, mid)

        merged.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": spk,
                "text": seg["text"],
            }
        )

    return merged


def coalesce_adjacent(items: List[Dict], gap_sec: float = 0.6) -> List[Dict]:
    if not items:
        return []

    items = sorted(items, key=lambda x: x["start"])
    out = [items[0].copy()]

    for cur in items[1:]:
        prev = out[-1]

        if (
            cur["speaker"] == prev["speaker"]
            and (cur["start"] - prev["end"]) <= gap_sec
        ):
            prev["end"] = max(prev["end"], cur["end"])
            prev["text"] = (prev["text"] + " " + cur["text"]).strip()
        else:
            out.append(cur.copy())

    return out


def format_txt(fragments: List[Dict]) -> str:
    lines = []

    for fr in fragments:
        lines.append(
            f"[{fr['start']:.2f} - {fr['end']:.2f}] {fr['speaker']}: {fr['text']}"
        )

    return "\n".join(lines)


def process_audio(
    input_path: str,
    enhance_mode: str = "none",
    asr_model: str = "large",
    device: str = "cuda",
) -> Tuple[List[Dict], str]:
    started = time.time()

    temp_root = None
    temp_files = []

    try:
        update_progress("Подготовка файла", 5, None, "processing")

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        temp_root = tempfile.mkdtemp(prefix="speech_core_")

        enhanced_wav = os.path.join(temp_root, f"{base_name}.enhanced.wav")
        mono16k_wav = os.path.join(temp_root, f"{base_name}.mono16k.wav")

        temp_files.extend([enhanced_wav, mono16k_wav])

        update_progress("Улучшение аудио", 12, 90, "processing")
        enhance_audio(input_path, enhanced_wav, mode=enhance_mode)

        update_progress("Конвертация в mono 16kHz", 20, 80, "processing")
        ensure_mono_16k(enhanced_wav, mono16k_wav)

        update_progress("Загрузка аудио", 28, 70, "processing")
        audio = load_wav_16k(mono16k_wav)

        update_progress("VAD: поиск речи", 38, 60, "processing")
        vad_segments = run_vad(audio)
        log.info("VAD segments: %d", len(vad_segments))

        if not vad_segments:
            update_progress("Готово", 100, 0, "done")
            return [], ""

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        if device != "cuda":
            raise RuntimeError("В этой версии используется только GPU diarization через NeMo.")

        update_progress("Диаризация спикеров", 58, 45, "processing")
        diar_segments = run_diarization_gpu(
            wav_path=mono16k_wav,
            vad_segments=vad_segments,
        )
        log.info("Diarization segments: %d", len(diar_segments))

        update_progress("ASR: распознавание речи", 80, 20, "processing")
        asr_segments = run_asr_over_vad(
            audio_16k=audio,
            vad_segments=vad_segments,
            device=device,
            model_size=asr_model,
        )
        log.info("ASR segments: %d", len(asr_segments))

        update_progress("Объединение результата", 92, 5, "processing")
        fragments = attach_speakers(asr_segments, diar_segments)
        fragments = coalesce_adjacent(fragments, gap_sec=0.6)

        result_text = format_txt(fragments)

        elapsed = time.time() - started
        log.info("Processing finished in %.2f sec", elapsed)

        update_progress("Готово", 100, 0, "done")
        return fragments, result_text

    except Exception:
        update_progress("Ошибка обработки", 100, 0, "error")
        raise

    finally:
        cleanup_paths(temp_files)
        if temp_root:
            cleanup_paths([temp_root])
