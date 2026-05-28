import os
import tempfile
from typing import Dict, List

import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering


_GPU_MODEL = None


def _get_gpu_model(model_name: str = "titanet_large"):
    global _GPU_MODEL

    if _GPU_MODEL is None:
        _GPU_MODEL = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=model_name
        )

        if torch.cuda.is_available():
            _GPU_MODEL = _GPU_MODEL.to("cuda")

        _GPU_MODEL.eval()

    return _GPU_MODEL


def _load_audio_mono16k(wav_path: str):
    audio, sr = sf.read(wav_path, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != 16000:
        raise ValueError(f"Expected 16k audio, got {sr} Hz")

    return audio, sr


def _normalize_embedding(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)

    if norm == 0:
        return x

    return x / norm


def _split_vad_segments(
    vad_segments: List[Dict],
    max_window_sec: float = 2.0,
    overlap_sec: float = 0.5,
    min_window_sec: float = 0.4,
) -> List[Dict]:
    windows = []

    for seg in vad_segments:
        start = float(seg["start"])
        end = float(seg["end"])
        dur = end - start

        if dur <= 0:
            continue

        if dur <= max_window_sec:
            if dur >= min_window_sec:
                windows.append({"start": start, "end": end})
            continue

        step = max_window_sec - overlap_sec
        cur = start

        while cur < end:
            win_end = min(cur + max_window_sec, end)

            if (win_end - cur) >= min_window_sec:
                windows.append({"start": cur, "end": win_end})

            if win_end >= end:
                break

            cur += step

    return windows


def _cluster_embeddings(
    embeddings: np.ndarray,
    distance_threshold: float = 0.48,
) -> np.ndarray:
    if len(embeddings) == 0:
        return np.array([], dtype=int)

    if len(embeddings) == 1:
        return np.array([0], dtype=int)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )

    labels = clustering.fit_predict(embeddings)

    return labels


def _reindex_labels(labels: np.ndarray) -> np.ndarray:
    unique = sorted(set(labels.tolist()))
    mapping = {old: new for new, old in enumerate(unique)}

    return np.array([mapping[x] for x in labels], dtype=int)


def _suppress_tiny_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int = 2,
) -> np.ndarray:
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

    big_clusters = [lab for lab, cnt in cluster_sizes.items() if cnt >= min_cluster_size]

    if not big_clusters:
        return labels

    centers = {}

    for lab in big_clusters:
        center = np.mean(embeddings[labels == lab], axis=0)
        centers[lab] = _normalize_embedding(center)

    new_labels = labels.copy()

    for lab, cnt in cluster_sizes.items():
        if cnt >= min_cluster_size:
            continue

        idxs = np.where(labels == lab)[0]

        for idx in idxs:
            emb = embeddings[idx]

            best_lab = None
            best_sim = -1.0

            for target_lab in big_clusters:
                sim = float(np.dot(emb, centers[target_lab]))

                if sim > best_sim:
                    best_sim = sim
                    best_lab = target_lab

            new_labels[idx] = best_lab

    return _reindex_labels(new_labels)


def _merge_similar_speakers(
    embeddings: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.78,
) -> np.ndarray:
    unique_labels = sorted(set(labels.tolist()))

    if len(unique_labels) <= 1:
        return labels

    speaker_vectors = {}

    for lab in unique_labels:
        vec = np.mean(embeddings[labels == lab], axis=0)
        speaker_vectors[lab] = _normalize_embedding(vec)

    parent = {lab: lab for lab in unique_labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)

        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(unique_labels):
        for b in unique_labels[i + 1:]:
            sim = float(np.dot(speaker_vectors[a], speaker_vectors[b]))

            if sim >= threshold:
                union(a, b)

    merged = np.array([find(x) for x in labels], dtype=int)

    return _reindex_labels(merged)


def _merge_adjacent_segments(
    items: List[Dict],
    gap_sec: float = 0.25,
) -> List[Dict]:
    if not items:
        return []

    items = sorted(items, key=lambda x: x["start"])
    merged = [items[0].copy()]

    for cur in items[1:]:
        prev = merged[-1]

        if (
            cur["speaker"] == prev["speaker"]
            and (cur["start"] - prev["end"]) <= gap_sec
        ):
            prev["end"] = max(prev["end"], cur["end"])
        else:
            merged.append(cur.copy())

    return merged


def run_diarization_gpu(
    wav_path: str,
    vad_segments: List[Dict],
    model_name: str = "titanet_large",
    max_window_sec: float = 3.0,
    overlap_sec: float = 0.5,
    min_window_sec: float = 0.4,
    distance_threshold: float = 0.40,
    merge_similarity_threshold: float = 0.65,
    min_cluster_size: int = 2,
    merge_gap_sec: float = 0.25,
) -> List[Dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for GPU diarization")

    audio, sr = _load_audio_mono16k(wav_path)
    model = _get_gpu_model(model_name=model_name)

    windows = _split_vad_segments(
        vad_segments=vad_segments,
        max_window_sec=max_window_sec,
        overlap_sec=overlap_sec,
        min_window_sec=min_window_sec,
    )

    if not windows:
        return []

    embeddings = []
    valid_windows = []

    for win in windows:
        s = int(win["start"] * sr)
        e = int(win["end"] * sr)
        chunk = audio[s:e]

        if len(chunk) < int(min_window_sec * sr):
            continue

        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            sf.write(tmp_path, chunk, sr)

            with torch.no_grad():
                emb = model.get_embedding(tmp_path)

            emb = emb.squeeze().detach().cpu().numpy().astype(np.float32)
            emb = _normalize_embedding(emb)

            embeddings.append(emb)
            valid_windows.append(win)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not embeddings:
        return [
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "speaker": "SPEAKER_00",
            }
            for seg in vad_segments
            if seg["end"] > seg["start"]
        ]

    embeddings = np.stack(embeddings, axis=0)

    labels = _cluster_embeddings(
        embeddings=embeddings,
        distance_threshold=distance_threshold,
    )

    labels = _suppress_tiny_clusters(
        embeddings=embeddings,
        labels=labels,
        min_cluster_size=min_cluster_size,
    )

    labels = _merge_similar_speakers(
        embeddings=embeddings,
        labels=labels,
        threshold=merge_similarity_threshold,
    )

    diar_segments = []

    for win, label in zip(valid_windows, labels):
        diar_segments.append(
            {
                "start": float(win["start"]),
                "end": float(win["end"]),
                "speaker": f"SPEAKER_{int(label):02d}",
            }
        )

    diar_segments = _merge_adjacent_segments(
        diar_segments,
        gap_sec=merge_gap_sec,
    )

    return diar_segments
