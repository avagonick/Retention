"""End-to-end local runner for TRIBE v2.

    python main.py path/to/video.mp4 [out_base] [--reuse]

Per run, creates  out_base/<video-name>/  and writes every artifact there:

    preds.npy                 raw (T, 20484) prediction array
    times.json                row start-times (sec)
    brain_static.png          2x2 cortical snapshot (most-active timestep)
    brain_interactive.html    rotatable 3D whole brain
    brain_activity.mp4        per-timestep animation (1 fps = real-time)
    networks_timeseries.png   per-network response (line + heatmap)
    networks.json             per-network time series
    engagement.png            engagement fitness curve + components
    engagement.json           LLM-ready payload (fitness, edit target, series)
    manifest.json             run summary (input, shape, fitness, file list)

Flow: GET /health -> POST /score -> local viz + network reduction + engagement.
out_base defaults to "out". --reuse re-renders from a cached preds.npy without
re-scoring (handy for viz/metric tweaks).

Credentials default to the CalHacks deployment; override with TRIBE_BASE_URL /
TRIBE_API_TOKEN.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

from config import BASE_URL, TOKEN
from tribe_client import check_health, score
import brain_viz
import brain_networks as bn
import engagement as eng


def run_name(video_path: str) -> str:
    """Filesystem-safe folder name derived from the video filename (no ext)."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "run"


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a for a in argv[1:] if a.startswith("--")}
    if not args:
        print("usage: python main.py path/to/video.mp4 [out_base] [--reuse]")
        return 2

    video_path = args[0]
    out_base = args[1] if len(args) > 1 else "out"
    reuse = "--reuse" in flags

    if not os.path.isfile(video_path):
        print(f"error: video not found: {video_path}")
        return 1

    run_dir = os.path.join(out_base, run_name(video_path))
    os.makedirs(run_dir, exist_ok=True)
    p = lambda name: os.path.join(run_dir, name)
    print(f"[run] {video_path}  ->  {run_dir}{os.sep}")

    # 1) Predictions: from cache (--reuse) or a fresh score.
    npy, tjson = p("preds.npy"), p("times.json")
    if reuse and os.path.isfile(npy) and os.path.isfile(tjson):
        preds = np.load(npy)
        times = json.load(open(tjson))
        print(f"[reuse] loaded cached predictions shape={preds.shape}")
    else:
        if not TOKEN:
            print("error: TRIBE_API_TOKEN not set — copy .env.example to .env and add your token.")
            return 1
        print(f"[health] GET {BASE_URL}/health ...")
        try:
            health = check_health(BASE_URL, TOKEN)
        except RuntimeError as exc:
            print(f"error: {exc}")
            return 1
        print(f"[health] {health}")
        if not health.get("model_loaded"):
            print("[health] note: model not loaded — first /score will cold-start (~2.5 min).")

        print(f"[score] uploading {video_path} (can take a few minutes) ...")
        t0 = time.time()
        try:
            preds, times = score(video_path, BASE_URL, TOKEN)
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"error: {exc}")
            return 1
        np.save(npy, preds)
        json.dump(times, open(tjson, "w"))
        print(
            f"[score] shape={preds.shape} T={len(times)} "
            f"window=[{times[0]:.0f}s..{times[-1]:.0f}s] "
            f"range=[{preds.min():.2f}, {preds.max():.2f}] in {time.time() - t0:.0f}s"
        )

    # 2) Cortical visualization.
    brain_viz.render_all(preds, times, prefix=p(""))

    # 3) Network reduction.
    print("[networks] reducing to 5 functional networks ...")
    bn.plot_network_timeseries(preds, times, p("networks_timeseries.png"))
    ts = bn.network_timeseries(preds)
    json.dump({k: v.tolist() for k, v in ts.items()}, open(p("networks.json"), "w"), indent=1)

    # 4) Engagement fitness.
    print("[engagement] computing fitness ...")
    result = eng.compute_engagement(preds, times)
    eng.plot_engagement(result, p("engagement.png"))
    payload = eng.to_payload(result)
    json.dump(payload, open(p("engagement.json"), "w"), indent=1)

    # 5) Run manifest.
    manifest = {
        "input_video": os.path.abspath(video_path),
        "run_dir": os.path.abspath(run_dir),
        "shape": list(preds.shape),
        "T": len(times),
        "overall_engagement": payload["overall_engagement"],
        "edit_target": payload["edit_target"],
        "coverage_end_s": payload["coverage_end_s"],
        "outputs": sorted(
            f for f in os.listdir(run_dir) if f != "manifest.json"
        ),
    }
    json.dump(manifest, open(p("manifest.json"), "w"), indent=1)

    print(f"\nDone -> {run_dir}{os.sep}")
    print(f"  overall engagement (fitness): {payload['overall_engagement']:+.4f}")
    print(f"  weakest moment (edit here):   stimulus {payload['edit_target']['t_stimulus']:.0f}s")
    for f in manifest["outputs"]:
        print(f"  - {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
