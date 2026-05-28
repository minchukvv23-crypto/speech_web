import os
import ffmpeg


def _to_wav(input_path: str, output_path: str, sr: int = 16000, mono: bool = True) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    kwargs = {
        "format": "wav",
        "acodec": "pcm_s16le",
        "ar": sr,
    }

    if mono:
        kwargs["ac"] = 1

    (
        ffmpeg
        .input(input_path)
        .output(output_path, **kwargs)
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path


def _medium_cleanup(input_path: str, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    stream = ffmpeg.input(input_path)
    audio = (
        stream.audio
        .filter("highpass", f=80)
        .filter("lowpass", f=8000)
        .filter("afftdn", nr=12)
        .filter("dynaudnorm", f=150, g=15)
    )

    (
        ffmpeg
        .output(
            audio,
            output_path,
            format="wav",
            acodec="pcm_s16le",
            ac=1,
            ar=16000,
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path


def _strong_cleanup(input_path: str, output_path: str) -> str:
    """
    Более сильная, но всё ещё осторожная обработка.
    Не делаем слишком агрессивный шумодав, чтобы не ломать речь.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    stream = ffmpeg.input(input_path)
    audio = (
        stream.audio
        .filter("highpass", f=70)
        .filter("lowpass", f=7600)
        .filter("afftdn", nr=18)
        .filter("dynaudnorm", f=90, g=21)
    )

    (
        ffmpeg
        .output(
            audio,
            output_path,
            format="wav",
            acodec="pcm_s16le",
            ac=1,
            ar=16000,
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path


def enhance_audio(input_path: str, output_path: str, mode: str = "none") -> str:
    if mode not in {"none", "medium", "strong"}:
        raise ValueError(f"Unsupported enhance mode: {mode}")

    if mode == "none":
        return _to_wav(input_path, output_path, sr=16000, mono=True)

    if mode == "medium":
        return _medium_cleanup(input_path, output_path)

    return _strong_cleanup(input_path, output_path)
