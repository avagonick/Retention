#!/usr/bin/env python3
"""Make a FAKE out/preds.npy so you can test the visualization without a GPU.

This produces the exact shape TRIBE v2 emits — (n_seconds, 20484) on the
fsaverage5 surface (10242 vertices/hemisphere) — filled with smooth, moving
hotspots. It is NOT real brain data; it just lets you confirm that
`make_brain_video.py` + ffmpeg render and sync correctly on your machine.

    python generate_synthetic_preds.py --seconds 52 --out out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PER_HEMI = 10242          # fsaverage5 vertices per hemisphere
N_VERTICES = 2 * PER_HEMI  # 20484


def hotspot(centers: np.ndarray, center: float, width: float) -> np.ndarray:
    """Gaussian bump over a 1-D vertex-index axis (fake but spatially smooth)."""
    return np.exp(-0.5 * ((centers - center) / width) ** 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=int, default=52, help="Number of TRs/frames.")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    idx = np.arange(N_VERTICES, dtype=np.float32)
    preds = np.zeros((args.seconds, N_VERTICES), dtype=np.float32)
    for t in range(args.seconds):
        phase = t / max(args.seconds - 1, 1)
        # One blob sweeps across the surface; a second pulses in/out. Gives an
        # obvious moving pattern so you can see the video is actually animating.
        blob_a = hotspot(idx, center=phase * N_VERTICES, width=900)
        blob_b = hotspot(idx, center=0.7 * N_VERTICES, width=600) * (0.5 + 0.5 *
                 np.sin(2 * np.pi * phase * 3))
        preds[t] = blob_a + 0.8 * blob_b + 0.02 * rng.standard_normal(N_VERTICES)

    np.save(out / "preds.npy", preds)
    # Fake captions so the overlay code path is exercised too.
    json.dump([f"synthetic frame {t}" for t in range(args.seconds)],
              open(out / "captions.json", "w"))
    print(f"Wrote {out/'preds.npy'}  shape={preds.shape}  (FAKE data)")
    print(f"Wrote {out/'captions.json'}")


if __name__ == "__main__":
    main()
