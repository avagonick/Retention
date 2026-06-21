"""Render TRIBE v2 predictions onto a cortical surface with nilearn.

All post-processing is local. Given a (T, 20484) prediction array, produces:

  * brain_static.png      — 2x2 (LEFT/RIGHT hemi x lateral/medial), most-active
                            timestep.
  * brain_interactive.html — rotatable 3D whole brain (view_surf).
  * brain_activity.mp4    — one 4-panel frame per timestep, animated over time.

Vertex layout (matches the model): columns 0..10241 = LEFT hemisphere,
10242..20483 = RIGHT hemisphere (10242 each, fsaverage5).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: render to buffers/files, never a window.

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
from nilearn.surface import load_surf_mesh

N_PER_HEMI = 10242

# (hemi, view) for the four panels, in subplot order.
_PANELS = [
    ("left", "lateral"),
    ("left", "medial"),
    ("right", "lateral"),
    ("right", "medial"),
]

_FSAVERAGE = None


def _fsaverage():
    """Fetch (and cache) the fsaverage5 surface meshes. Auto-downloads once."""
    global _FSAVERAGE
    if _FSAVERAGE is None:
        _FSAVERAGE = datasets.fetch_surf_fsaverage("fsaverage5")
    return _FSAVERAGE


def _split(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 20484-vector into (left, right) hemisphere halves."""
    return row[:N_PER_HEMI], row[N_PER_HEMI:]


def _scale(preds: np.ndarray) -> float:
    """Symmetric color scale, stable across frames (97th pct of |activity|)."""
    vmax = float(np.percentile(np.abs(preds), 97))
    # Guard the degenerate all-near-zero case so threshold/vmax stay valid.
    return vmax if vmax > 1e-6 else 1.0


def most_active_timestep(preds: np.ndarray) -> int:
    """Index of the timestep with the largest mean absolute activity."""
    return int(np.argmax(np.abs(preds).mean(axis=1)))


def _panel_figure(
    row: np.ndarray,
    vmax: float,
    fs,
    *,
    figsize=(11.2, 9.6),  # x100 dpi -> 1120x960, divisible by 16 (clean mp4 frames)
    dpi=100,
    title: str | None = None,
):
    """Build the 4-panel (hemi x view) figure for a single timestep's row."""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    for i, (hemi, view) in enumerate(_PANELS):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        dat = _split(row)[0 if hemi == "left" else 1]
        plotting.plot_surf_stat_map(
            fs[f"infl_{hemi}"],
            stat_map=dat,
            hemi=hemi,
            view=view,
            bg_map=fs[f"sulc_{hemi}"],
            bg_on_data=True,
            cmap="cold_hot",
            vmax=vmax,
            threshold=vmax * 0.15,
            colorbar=(i == 3),
            axes=ax,
            figure=fig,
        )
        ax.set_title(f"{hemi} · {view}", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=13)
    return fig


def render_static(preds: np.ndarray, times, out_path="brain_static.png") -> str:
    """2x2 figure for the most-active timestep. Returns the output path."""
    fs = _fsaverage()
    vmax = _scale(preds)
    t = most_active_timestep(preds)
    title = f"Most-active timestep  t={t}  ({times[t]:.0f}s)"
    fig = _panel_figure(preds[t], vmax, fs, title=title)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_interactive(
    preds: np.ndarray, times, out_path="brain_interactive.html"
) -> str:
    """Rotatable 3D whole brain (view_surf) for the most-active timestep."""
    fs = _fsaverage()
    vmax = _scale(preds)
    t = most_active_timestep(preds)

    # Merge the two inflated hemispheres into one mesh, nudged apart on x so
    # they don't overlap, then map the full 20484-vector onto it.
    lc, lf = load_surf_mesh(fs["infl_left"])
    rc, rf = load_surf_mesh(fs["infl_right"])
    lc = lc.copy()
    lc[:, 0] -= lc[:, 0].max() + 2
    rc = rc.copy()
    rc[:, 0] -= rc[:, 0].min() - 2
    mesh = (np.vstack([lc, rc]), np.vstack([lf, rf + lc.shape[0]]))

    view = plotting.view_surf(
        mesh,
        surf_map=preds[t],
        cmap="cold_hot",
        symmetric_cmap=True,
        vmax=vmax,
        threshold=vmax * 0.15,
    )
    view.save_as_html(out_path)
    return out_path


def render_video(
    preds: np.ndarray, times, out_path="brain_activity.mp4", fps=1
) -> str:
    """One 4-panel frame per timestep, animated over time. Returns the path.

    fps=1 plays in lockstep with the source clip (one frame per 1-second brain
    window); raise it for a faster-than-real-time animation.
    """
    fs = _fsaverage()
    vmax = _scale(preds)  # global scale -> brightness comparable across frames
    n = preds.shape[0]

    frames = []
    for t in range(n):
        title = f"t={t}  ({times[t]:.0f}s)   [{t + 1}/{n}]"
        fig = _panel_figure(preds[t], vmax, fs, title=title)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(frame)
        plt.close(fig)
        print(f"  frame {t + 1}/{n}", end="\r", flush=True)
    print()

    imageio.mimsave(out_path, frames, fps=fps)
    return out_path


def render_all(preds: np.ndarray, times, *, prefix: str = "") -> dict:
    """Render all three outputs. Returns a dict of {name: path}.

    `prefix` is prepended to each filename (e.g. an output directory + os.sep,
    or a run id).
    """
    outputs = {}
    print("[viz] static PNG ...")
    outputs["static"] = render_static(preds, times, f"{prefix}brain_static.png")
    print("[viz] interactive HTML ...")
    outputs["interactive"] = render_interactive(
        preds, times, f"{prefix}brain_interactive.html"
    )
    print("[viz] activity MP4 ...")
    outputs["video"] = render_video(preds, times, f"{prefix}brain_activity.mp4")
    return outputs
