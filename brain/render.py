"""Render TRIBE v2 predictions onto a cortical surface with nilearn.

All post-processing is local. Given a (T, 20484) prediction array, produces:

  * brain_static.png      â€” 2x2 (LEFT/RIGHT hemi x lateral/medial), most-active
                            timestep.
  * brain_interactive.html â€” rotatable 3D whole brain (view_surf).
  * brain_activity.mp4    â€” one 4-panel frame per timestep, animated over time.

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

# Cortical surface mesh to render onto. "pial" = realistic folded brain.
# (other fsaverage5 options: "infl", "white", "flat", "sphere")
SURFACE = "infl"

# Mesh we *render* on. "fsaverage5" = native data resolution (no upsampling).
# Bump to "fsaverage6"/"fsaverage7" to interpolate onto a finer mesh.
RENDER_MESH = "fsaverage5"

# Hide activity below this fraction of vmax.
THRESHOLD_FRAC = 0.15

# Background colour for the figure / panels.
BG_COLOR = "black"

# Hemodynamic lag (sec) â€” keep in sync with retention.DEFAULT_LAG_S. Used to
# shift the animation so the brain response aligns with the on-screen moment.
LAG_S = 5.0

# (hemi, view) for the four panels, in subplot order.
_PANELS = [
    ("left", "lateral"),
    ("left", "medial"),
    ("right", "lateral"),
    ("right", "medial"),
]

_MESHES = None


def _meshes():
    """Fetch (cache) the hi-res render mesh + an fsaverage5->render interpolator.

    fsaverage meshes are nested icosahedra, so a few-nearest-neighbour inverse-
    distance interpolation on the sphere upsamples fsaverage5 data cleanly onto
    the higher-res mesh (exact at the shared vertices, smooth in between).
    """
    global _MESHES
    if _MESHES is None:
        from scipy.spatial import cKDTree

        hi = datasets.fetch_surf_fsaverage(RENDER_MESH)
        lo = datasets.fetch_surf_fsaverage("fsaverage5")
        interp = {}
        for hemi in ("left", "right"):
            lo_sph, _ = load_surf_mesh(lo[f"sphere_{hemi}"])
            hi_sph, _ = load_surf_mesh(hi[f"sphere_{hemi}"])
            dist, idx = cKDTree(lo_sph).query(hi_sph, k=3)
            w = 1.0 / np.maximum(dist, 1e-6)
            w /= w.sum(axis=1, keepdims=True)
            interp[hemi] = (idx, w)
        _MESHES = {"hi": hi, "interp": interp}
    return _MESHES


def _upsample(vals: np.ndarray, hemi: str) -> np.ndarray:
    """Interpolate a (10242,) fsaverage5 hemi vector onto the render mesh."""
    if RENDER_MESH == "fsaverage5":  # native resolution, no interpolation
        return vals
    idx, w = _meshes()["interp"][hemi]
    return (vals[idx] * w).sum(axis=1)


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
    *,
    figsize=(11.2, 9.6),  # x100 dpi -> 1120x960, divisible by 16 (clean mp4 frames)
    dpi=100,
    title: str | None = None,
):
    """Build the 4-panel (hemi x view) figure for a single timestep's row."""
    fs = _meshes()["hi"]
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=BG_COLOR)
    for i, (hemi, view) in enumerate(_PANELS):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.set_facecolor(BG_COLOR)
        dat = _upsample(_split(row)[0 if hemi == "left" else 1], hemi)
        plotting.plot_surf_stat_map(
            fs[f"{SURFACE}_{hemi}"],
            stat_map=dat,
            hemi=hemi,
            view=view,
            bg_map=fs[f"sulc_{hemi}"],
            bg_on_data=True,
            cmap="cold_hot",
            vmax=vmax,
            threshold=vmax * THRESHOLD_FRAC,
            colorbar=(i == 3),
            axes=ax,
            figure=fig,
        )
        ax.set_title(f"{hemi} Â· {view}", fontsize=10, color="white")
    if title:
        fig.suptitle(title, fontsize=13, color="white")
    # Whiten colorbar ticks/labels so they read on the black background.
    for cax in fig.axes:
        cax.tick_params(colors="white")
    return fig


def render_static(preds: np.ndarray, times, out_path="brain_static.png") -> str:
    """2x2 figure for the most-active timestep. Returns the output path."""
    vmax = _scale(preds)
    t = most_active_timestep(preds)
    title = f"Most-active timestep  t={t}  ({times[t]:.0f}s)"
    fig = _panel_figure(preds[t], vmax, title=title)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    return out_path


def render_interactive(
    preds: np.ndarray, times, out_path="brain_interactive.html"
) -> str:
    """Rotatable 3D whole brain (view_surf) for the most-active timestep."""
    fs = _meshes()["hi"]
    vmax = _scale(preds)
    t = most_active_timestep(preds)

    # Merge the two hemispheres into one mesh, nudged apart on x so they don't
    # overlap, then map the upsampled activity (both hemis) onto it.
    surf_map = np.concatenate([
        _upsample(_split(preds[t])[0], "left"),
        _upsample(_split(preds[t])[1], "right"),
    ])
    lc, lf = load_surf_mesh(fs[f"{SURFACE}_left"])
    rc, rf = load_surf_mesh(fs[f"{SURFACE}_right"])
    lc = lc.copy()
    lc[:, 0] -= lc[:, 0].max() + 2
    rc = rc.copy()
    rc[:, 0] -= rc[:, 0].min() - 2
    mesh = (np.vstack([lc, rc]), np.vstack([lf, rf + lc.shape[0]]))

    view = plotting.view_surf(
        mesh,
        surf_map=surf_map,
        cmap="cold_hot",
        symmetric_cmap=True,
        vmax=vmax,
        threshold=vmax * THRESHOLD_FRAC,
    )
    view.save_as_html(out_path)
    return out_path


def render_video(
    preds: np.ndarray,
    times,
    out_path="brain_activity.mp4",
    fps=1,
    lag_s: float = LAG_S,
    sync_to_stimulus: bool = True,
) -> str:
    """One 4-panel frame per timestep, animated over time. Returns the path.

    fps=1 plays one frame per 1-second window. With sync_to_stimulus=True
    (default) the animation is shifted by the hemodynamic lag so the brain's
    response lines up with the on-screen moment that caused it: pre-video
    ramp-up frames (stimulus-time = brain_time - lag_s < 0) are dropped, and
    each frame is labelled by video/stimulus time. Set False for the raw
    brain-time animation.
    """
    vmax = _scale(preds)  # global scale over ALL rows -> comparable brightness
    times = np.asarray(times, dtype=float)
    n = preds.shape[0]

    idxs = [t for t in range(n) if times[t] - lag_s >= 0] if sync_to_stimulus else list(range(n))
    if not idxs:  # clip shorter than the lag â€” fall back to the full timeline
        idxs = list(range(n))
        sync_to_stimulus = False
    m = len(idxs)

    frames = []
    for k, t in enumerate(idxs):
        if sync_to_stimulus:
            title = f"video {times[t] - lag_s:.0f}s   Â·   brain {times[t]:.0f}s   [{k + 1}/{m}]"
        else:
            title = f"brain {times[t]:.0f}s   [{k + 1}/{m}]"
        fig = _panel_figure(preds[t], vmax, title=title)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(frame)
        plt.close(fig)
        print(f"  frame {k + 1}/{m}", end="\r", flush=True)
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
