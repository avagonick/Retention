#!/usr/bin/env python3
"""Stage 1 — run TRIBE v2 on a video and save its predicted brain activity.

TRIBE v2 (Meta FAIR, https://github.com/facebookresearch/tribev2) predicts the
fMRI response of an "average" human brain to naturalistic stimuli. Given a video
it outputs one cortical activity map per second (1 TR = 1 s) on the **fsaverage5**
surface (~20k vertices). The predictions are already shifted 5 s into the past to
compensate for the hemodynamic lag, so prediction frame `i` lines up with second
`i` of the stimulus.

This stage is the heavy one (loads V-JEPA2 + Wav2Vec-BERT + LLaMA-3.2 + DINOv2 +
the TRIBE transformer). Run it on a GPU box / Colab. It writes two small files
that Stage 2 (`make_brain_video.py`) consumes:

    out/preds.npy       float32 array, shape (n_seconds, ~20484)
    out/captions.json   list[str], the transcribed words shown at each second

Usage:
    python run_inference.py --video cache/sample_video.mp4 --out out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="cache/sample_video.mp4",
                    help="Path to the stimulus video (.mp4/.mov/.mkv/...).")
    ap.add_argument("--out", default="out", help="Output directory.")
    ap.add_argument("--cache", default="cache",
                    help="HuggingFace / checkpoint cache folder.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Imported lazily so `--help` works without the heavy deps installed.
    from tribev2.demo_utils import TribeModel
    from tribev2.plotting.utils import get_text

    print("Loading TRIBE v2 from HuggingFace (facebook/tribev2)...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache)

    print(f"Extracting multimodal events from {args.video} ...")
    df = model.get_events_dataframe(video_path=args.video)

    print("Running inference (this is the slow part)...")
    preds, segments = model.predict(events=df)  # (n_seconds, n_vertices)
    preds = np.asarray(preds, dtype=np.float32)
    print(f"  predictions: {preds.shape}  (n_seconds, n_vertices)")

    # Per-second caption text (what the model "heard"), for the video overlay.
    captions = [get_text(s) if s is not None else "" for s in segments]

    np.save(out / "preds.npy", preds)
    (out / "captions.json").write_text(json.dumps(captions))
    print(f"Saved {out/'preds.npy'} and {out/'captions.json'}")
    print("Next: python make_brain_video.py --video", args.video, "--out", args.out)


if __name__ == "__main__":
    main()
