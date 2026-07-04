# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Patched for environments where torchcodec is unavailable (missing libnppicc).
# Falls back to PyAV (pip install av) which works everywhere.

import math
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Backend selection: try torchcodec first, fall back to PyAV
# ---------------------------------------------------------------------------

def _try_torchcodec():
    try:
        from torchcodec.decoders import VideoDecoder # noqa: F401
        VideoDecoder # trigger import so OSError surfaces now
        return True
    except Exception:
        return False


_USE_TORCHCODEC = _try_torchcodec()


# ---------------------------------------------------------------------------
# PyAV fallback helpers
# ---------------------------------------------------------------------------

def _av_decode_indices(video_path: str, indices: list) -> np.ndarray:
    """Decode specific frame indices using PyAV."""
    import av
    indices_set = set(indices)
    frames_map = {}
    container = av.open(video_path)
    stream = container.streams.video[0]
    for i, frame in enumerate(container.decode(stream)):
        if i in indices_set:
            frames_map[i] = frame.to_ndarray(format="rgb24")
        if len(frames_map) == len(indices_set):
            break
    container.close()
    # Return in requested order, shape (N, H, W, 3)
    result = np.stack([frames_map[i] for i in indices])
    return result


def _av_decode_timestamps(video_path: str, timestamps: np.ndarray) -> np.ndarray:
    """Decode frames nearest to given timestamps using PyAV."""
    import av
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    all_frames = []
    all_pts = []
    for frame in container.decode(stream):
        all_frames.append(frame.to_ndarray(format="rgb24"))
        t = frame.pts * stream.time_base if frame.pts is not None else len(all_pts) / fps
        all_pts.append(float(t))
    container.close()
    all_pts = np.array(all_pts)
    # For each requested timestamp, find nearest frame
    indices = [int(np.argmin(np.abs(all_pts - t))) for t in timestamps]
    return np.stack([all_frames[i] for i in indices])


def _av_decode_all(video_path: str):
    """Decode all frames using PyAV. Returns (frames, pts_seconds)."""
    import av
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    frames, pts = [], []
    for i, frame in enumerate(container.decode(stream)):
        frames.append(frame.to_ndarray(format="rgb24"))
        t = frame.pts * stream.time_base if frame.pts is not None else i / fps
        pts.append(float(t))
    container.close()
    return np.stack(frames), np.array(pts)


# ---------------------------------------------------------------------------
# Public API (same signatures as original)
# ---------------------------------------------------------------------------

_DEFAULT_DECODER_KWARGS: dict = {
    "device": "cpu",
    "dimension_order": "NHWC",
    "num_ffmpeg_threads": 0,
}


def get_frames_by_indices(
    video_path: str,
    indices: "list[int] | np.ndarray",
    decoder_kwargs: Optional[dict] = None,
) -> np.ndarray:
    if _USE_TORCHCODEC:
        from torchcodec.decoders import VideoDecoder
        kwargs = {**_DEFAULT_DECODER_KWARGS, **(decoder_kwargs or {})}
        dec = VideoDecoder(video_path, **kwargs)
        return dec.get_frames_at(indices=indices).data.numpy()
    else:
        return _av_decode_indices(video_path, list(indices))


def get_frames_by_timestamps(
    video_path: str,
    timestamps: "list[float] | np.ndarray",
    decoder_kwargs: Optional[dict] = None,
) -> np.ndarray:
    timestamps = np.array(timestamps, dtype=np.float64)
    if _USE_TORCHCODEC:
        from torchcodec.decoders import VideoDecoder
        kwargs = {**_DEFAULT_DECODER_KWARGS, **(decoder_kwargs or {})}
        dec = VideoDecoder(video_path, **kwargs)
        fps = dec.metadata.average_fps
        interval = 1.0 / fps
        closest = np.round(timestamps / interval) * interval
        errors = np.abs(closest - timestamps) / interval
        if np.any(errors >= 0.01):
            bad = timestamps[errors >= 0.01]
            raise ValueError(
                f"Invalid timestamps {bad} for video {video_path} (FPS: {fps})"
            )
        return dec.get_frames_played_at(seconds=closest).data.numpy()
    else:
        return _av_decode_timestamps(video_path, timestamps)


def get_all_frames(
    video_path: str,
    decoder_kwargs: Optional[dict] = None,
) -> "tuple[np.ndarray, np.ndarray]":
    if _USE_TORCHCODEC:
        from torchcodec.decoders import VideoDecoder
        kwargs = {**_DEFAULT_DECODER_KWARGS, **(decoder_kwargs or {})}
        dec = VideoDecoder(video_path, **kwargs)
        frames = dec.get_frames_at(indices=range(len(dec)))
        return frames.data.numpy(), frames.pts_seconds.numpy()
    else:
        return _av_decode_all(video_path)


def get_accumulate_timestamp_idxs(
    timestamps: List[float],
    start_time: float,
    dt: float,
    eps: float = 1e-5,
    next_global_idx: Optional[int] = 0,
    allow_negative=False,
) -> Tuple[List[int], List[int], int]:
    local_idxs: list = []
    global_idxs: list = []
    for local_idx, ts in enumerate(timestamps):
        global_idx = math.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx
        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats
    return local_idxs, global_idxs, next_global_idx
