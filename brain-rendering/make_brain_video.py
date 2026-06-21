#!/usr/bin/env python3
"""Stage 2 — turn TRIBE's predicted activity into a watchable brain video.

Reads the `preds.npy` / `captions.json` produced by `run_inference.py` and renders
one cortical-surface frame per second, then uses ffmpeg to build two outputs:

    out/brain.mp4          the brain alone, lighting up over time
    out/side_by_side.mp4   the stimulus video next to the brain, time-synced

Why they line up: TRIBE emits one map per second (1 TR = 1 s) and already shifts
its predictions 5 s into the past to undo the hemodynamic lag, so brain frame `i`
is the response to second `i` of the stimulus. A plain ffmpeg hstack therefore
keeps them aligned.

This stage is light (matplotlib on CPU) and runs fine on a laptop. It does NOT
load the model, so you can iterate on the visualization without a GPU.

Usage:
    python make_brain_video.py --video cache/sample_video.mp4 --out out
    python make_brain_video.py --out out --backend pyvista   # nicer 3D, needs OpenGL
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def render_frames(preds, captions, frames_dir: Path, backend: str) -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    # Nilearn backend = pure matplotlib, portable (no GPU/OpenGL). PyVista looks
    # nicer in 3D but needs an OpenGL context, which is painful headless.
    if backend == "pyvista":
        from tribev2.plotting import PlotBrainPyvista as Plotter
    else:
        from tribev2.plotting import PlotBrainNilearn as Plotter
    plotter = Plotter(mesh="fsaverage5")

    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("frame_*.png"):
        f.unlink()

    # Same look as the official demo notebook: "fire" colormap, top-of-range only.
    # "left" = lateral view of the left hemisphere (visual cortex posterior,
    # language network along the temporal lobe). See VIEW_DICT for other angles.
    style = dict(cmap="fire", norm_percentile=99, vmin=0.6, alpha_cmap=(0, 0.2),
                 views="left")
    for i in tqdm(range(len(preds)), desc="Rendering brain frames"):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        plotter.plot_surf(preds[i], axes=[ax], **style)
        fig.suptitle(f"t = {i}s", fontsize=13, fontweight="bold")
        if i < len(captions) and captions[i]:
            words = " ".join(captions[i].split(" ")[-8:])
            fig.text(0.5, 0.04, words, fontsize=9, ha="center", va="bottom",
                     wrap=True)
        fig.savefig(frames_dir / f"frame_{i:05d}.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_videos(frames_dir: Path, out: Path, stimulus: Path | None,
                 fps_smooth: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found. Install it (macOS: `brew install ffmpeg`).")

    brain = out / "brain.mp4"
    # 1 frame per second, optionally motion-interpolated to look smooth.
    cmd = ["ffmpeg", "-y", "-framerate", "1",
           "-i", str(frames_dir / "frame_%05d.png")]
    # trunc(iw/2)*2 rounds width/height down to the nearest even number —
    # libx264 requires even dimensions.
    even = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    vf = f"{even},minterpolate=fps={fps_smooth}" if fps_smooth else even
    cmd += ["-vf", vf, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(brain)]
    run(cmd)
    print(f"Wrote {brain}")

    if stimulus and stimulus.exists():
        side = out / "side_by_side.mp4"
        # Scale both to height 512 and place stimulus | brain. -shortest trims to
        # whichever ends first (they should be ~equal: 1 TR == 1 s of stimulus).
        fc = ("[0:v]scale=-2:512,setsar=1[a];"
              f"[1:v]scale=-2:512,setsar=1,fps={fps_smooth or 1}[b];"
              "[a][b]hstack=inputs=2[v]")
        run(["ffmpeg", "-y", "-i", str(stimulus), "-i", str(brain),
             "-filter_complex", fc, "-map", "[v]", "-map", "0:a?",
             "-shortest", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
             str(side)])
        print(f"Wrote {side}")
    else:
        print("No stimulus video given/found — skipped side_by_side.mp4")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out", help="Dir with preds.npy / outputs.")
    ap.add_argument("--video", default="cache/sample_video.mp4",
                    help="Stimulus video for the side-by-side (optional).")
    ap.add_argument("--backend", choices=["nilearn", "pyvista"], default="nilearn")
    ap.add_argument("--smooth-fps", type=int, default=30,
                    help="Motion-interpolate the 1 fps brain to this fps. 0 = off.")
    args = ap.parse_args()

    out = Path(args.out)
    preds = np.load(out / "preds.npy")
    captions_path = out / "captions.json"
    captions = json.loads(captions_path.read_text()) if captions_path.exists() else []
    print(f"Loaded predictions {preds.shape} from {out/'preds.npy'}")

    frames_dir = out / "frames"
    render_frames(preds, captions, frames_dir, args.backend)
    build_videos(frames_dir, out, Path(args.video), args.smooth_fps)


if __name__ == "__main__":
    main()
