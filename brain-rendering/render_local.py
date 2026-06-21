#!/usr/bin/env python3
"""Lightweight, torch-free renderer for TRIBE v2 predictions.

`make_brain_video.py` imports the full `tribev2` package, which pulls in torch +
neuralset (won't install on Python 3.14, and is heavy). This script does the same
job — render `out/preds.npy` onto the fsaverage5 cortex and stitch a video — using
only **nilearn + matplotlib + ffmpeg**, so it runs on a laptop with no GPU.

It relies on the same convention TRIBE uses: a 20484-vector is [left 10242 |
right 10242] on fsaverage5 (verified in tribev2/plotting/base.py:get_stat_map).

    pip install nilearn matplotlib
    brew install ffmpeg            # macOS
    python render_local.py --out out --video cache/sample_video.mp4

Outputs out/brain.mp4 and (if a stimulus is given) out/side_by_side.mp4.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

PER_HEMI = 10242  # fsaverage5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out")
    ap.add_argument("--video", default="cache/sample_video.mp4",
                    help="Stimulus for the side-by-side (optional).")
    ap.add_argument("--view", default="lateral",
                    help="nilearn view: lateral, medial, dorsal, ventral, ...")
    ap.add_argument("--hemi", default="left", choices=["left", "right"])
    ap.add_argument("--surface", default="pial", choices=["pial", "infl"],
                    help="pial = real folded brain (default); infl = inflated.")
    ap.add_argument("--cmap", default="turbo",
                    help="Activation gradient colormap (turbo, inferno, plasma, ...).")
    ap.add_argument("--smooth-fps", type=int, default=30, help="0 to disable.")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting
    from tqdm import tqdm

    out = Path(args.out)
    preds = np.load(out / "preds.npy")
    if preds.shape[1] != 2 * PER_HEMI:
        raise SystemExit(f"Expected {2*PER_HEMI} vertices (fsaverage5), "
                         f"got {preds.shape[1]}.")
    cap_path = out / "captions.json"
    captions = json.loads(cap_path.read_text()) if cap_path.exists() else []
    print(f"Loaded {preds.shape} from {out/'preds.npy'}")

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    # pial = the real folded cortical surface; use infl_* for the inflated balloon.
    mesh = fs[f"{args.surface}_{args.hemi}"]
    bg = fs[f"sulc_{args.hemi}"]
    sl = slice(0, PER_HEMI) if args.hemi == "left" else slice(PER_HEMI, None)

    # Consistent brightness across time: one vmax from the 99th pct of all frames;
    # threshold hides the grey background so only hotspots glow (the "fire" look).
    vmax = float(np.percentile(preds, 99)) or 1.0
    thr = 0.45 * vmax

    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    for f in frames.glob("frame_*.png"):
        f.unlink()

    for i in tqdm(range(len(preds)), desc="Rendering"):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(4, 4))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        plotting.plot_surf_stat_map(
            mesh, preds[i, sl], hemi=args.hemi, view=args.view, bg_map=bg,
            cmap=args.cmap, threshold=thr, vmax=vmax, colorbar=False,
            bg_on_data=True, axes=ax, figure=fig,
        )
        fig.suptitle(f"t = {i}s", fontsize=13, fontweight="bold", color="white")
        if i < len(captions) and captions[i]:
            fig.text(0.5, 0.04, captions[i], fontsize=9, ha="center", va="bottom",
                     color="white")
        fig.savefig(frames / f"frame_{i:05d}.png", dpi=150, facecolor="black")
        plt.close(fig)

    if shutil.which("ffmpeg") is None:
        raise SystemExit("Frames rendered, but ffmpeg not found "
                         "(macOS: `brew install ffmpeg`) — can't make the mp4.")

    brain = out / "brain.mp4"
    cmd = ["ffmpeg", "-y", "-framerate", "1", "-i", str(frames / "frame_%05d.png")]
    if args.smooth_fps:
        cmd += ["-vf", f"minterpolate=fps={args.smooth_fps}"]
    cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(brain)]
    print("+", " ".join(cmd)); subprocess.run(cmd, check=True)
    print(f"Wrote {brain}")

    stim = Path(args.video)
    if stim.exists():
        side = out / "side_by_side.mp4"
        fc = ("[0:v]scale=-2:512,setsar=1[a];"
              f"[1:v]scale=-2:512,setsar=1,fps={args.smooth_fps or 1}[b];"
              "[a][b]hstack=inputs=2[v]")
        cmd = ["ffmpeg", "-y", "-i", str(stim), "-i", str(brain),
               "-filter_complex", fc, "-map", "[v]", "-map", "0:a?",
               "-shortest", "-c:v", "libx264", "-crf", "18",
               "-pix_fmt", "yuv420p", str(side)]
        print("+", " ".join(cmd)); subprocess.run(cmd, check=True)
        print(f"Wrote {side}")
    else:
        print(f"No stimulus at {stim} — skipped side_by_side.mp4")


if __name__ == "__main__":
    main()
