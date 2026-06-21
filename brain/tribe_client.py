"""TRIBE v2 remote client.

Thin HTTP wrapper around the TRIBE v2 neuroscience model served on a
Lightning AI GPU. Upload an .mp4, get back predicted brain activity as a
(T, 20484) float32 array (z-scored fMRI BOLD over fsaverage5 cortical
vertices).

Vertex layout: columns 0..10241 = LEFT hemisphere, 10242..20483 = RIGHT.
Rows = ~1-second timesteps; T is variable (always read it from the response,
never hard-code it).
"""

from __future__ import annotations

import os

import numpy as np
import requests

# Number of cortical vertices the model always returns (fsaverage5, both hemis).
N_VERTICES = 20484
N_PER_HEMI = 10242

# Default scoring timeout. The first call after the server cold-starts triggers
# a lazy model load (~2.5 min), so this must stay generous (>= 600 s).
DEFAULT_TIMEOUT = 600


def _join(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def check_health(base_url: str, token: str, timeout: int = 30) -> dict:
    """GET /health — confirm connectivity and that the model is loaded.

    Returns the parsed JSON, e.g.::

        {"status": "ok", "model_loaded": true,
         "n_vertices": 20484, "hemisphere_split": 10242}

    Raises a clear error on any non-200 response.
    """
    url = _join(base_url, "health")
    headers = {"X-API-Token": token}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"TRIBE /health unreachable at {url}: {exc}") from exc
    if resp.status_code != 200:
        raise RuntimeError(
            f"TRIBE /health returned {resp.status_code}: {resp.text[:500]}"
        )
    return resp.json()


def score(
    video_path: str,
    base_url: str,
    token: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[np.ndarray, list[float]]:
    """POST an .mp4 to /score and return (preds, times).

    Parameters
    ----------
    video_path : path to the .mp4 to score.
    base_url   : remote API base URL.
    token      : value for the X-API-Token header.
    timeout    : client timeout in seconds (default 600; keep >= 600 for the
                 cold-start lazy model load).

    Returns
    -------
    preds : np.ndarray, shape (T, 20484), dtype float32 — predicted activity.
    times : list[float], length T — start time (sec) of each row's 1-sec window.

    Raises
    ------
    FileNotFoundError if the video is missing.
    RuntimeError on any non-200 response or malformed payload.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"video not found: {video_path}")

    url = _join(base_url, "score")
    headers = {
        "X-API-Token": token,
        # Ask the server to gzip the (~6.7 MB) body; requests transparently
        # decompresses on the way back.
        "Accept-Encoding": "gzip",
    }

    with open(video_path, "rb") as fh:
        files = {"file": (os.path.basename(video_path), fh, "video/mp4")}
        try:
            resp = requests.post(
                url, headers=headers, files=files, timeout=timeout
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"TRIBE /score request failed: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"TRIBE /score returned {resp.status_code}: {resp.text[:500]}"
        )

    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"TRIBE /score returned non-JSON ({len(resp.content)} bytes): "
            f"{resp.text[:200]}"
        ) from exc

    for key in ("shape", "dtype", "times", "data"):
        if key not in payload:
            raise RuntimeError(
                f"TRIBE /score payload missing key '{key}'; got keys "
                f"{sorted(payload)}"
            )

    times = list(payload["times"])
    preds = np.asarray(payload["data"], dtype=np.float32)

    # Validate against the model's stated conventions rather than trusting blindly.
    if preds.ndim != 2:
        raise RuntimeError(f"expected 2-D data, got shape {preds.shape}")
    if preds.shape[1] != N_VERTICES:
        raise RuntimeError(
            f"expected {N_VERTICES} vertices per row, got {preds.shape[1]}"
        )
    if preds.shape[0] != len(times):
        raise RuntimeError(
            f"row count {preds.shape[0]} != len(times) {len(times)}"
        )

    declared = tuple(payload["shape"])
    if declared != preds.shape:
        raise RuntimeError(
            f"declared shape {declared} != parsed array shape {preds.shape}"
        )

    return preds, times
