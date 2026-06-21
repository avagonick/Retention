"""Collapse the 20,484-vertex TRIBE array into named functional networks.

This is the *engagement-reduction* layer: it turns the raw per-vertex
prediction grid into a small, interpretable set of per-network time series
(visual, auditory, language, frontoparietal/attention, default-mode) that the
LLM step can reason over.

Atlas: Destrieux 2009 surface parcellation on fsaverage5 (nilearn,
`fetch_atlas_surf_destrieux`). Anatomical, but its fine regions (Heschl's
gyrus, IFG, intraparietal sulcus, precuneus, ...) map cleanly onto the five
named networks — which the coarse Yeo-7 functional atlas does NOT (it has no
distinct auditory or language network). The mapping below is keyword-based but
materialized explicitly, so it is fully auditable and easy to edit.

The *engagement score* (how to combine these five series into one number) is a
separate, deliberate design decision and lives in `engagement.py`, not here.
"""

from __future__ import annotations

import numpy as np
from nilearn import datasets

N_PER_HEMI = 10242

# The five named networks, in display order.
NETWORKS = ["visual", "auditory", "language", "frontoparietal", "default_mode"]

# Ordered keyword rules. Each Destrieux region name is assigned to the FIRST
# network whose keyword it contains; unmatched regions (sensorimotor, insula,
# etc.) fall through to "other" and are excluded from the named networks.
# Order matters — visual/auditory are matched before language so planum/Heschl
# go to auditory and parahippocampal goes to default-mode, not visual.
_RULES: list[tuple[str, list[str]]] = [
    ("visual", [
        "occipital", "G_cuneus", "calcarine", "Lingual", "fusifor", "Lunatus",
        "parieto_occipital", "oc_sup", "oc_middle", "oc-temp_lat",
        "collat_transv", "Pole_occipital",
    ]),
    ("auditory", [
        "T_transv", "Plan_polar", "Plan_tempo", "temporal_transverse",
    ]),
    ("language", [
        "front_inf-Opercular", "front_inf-Triangul", "temp_sup-Lateral",
        "temporal_sup", "temporal_middle", "Supramar", "Lat_Fis-post",
    ]),
    ("default_mode", [
        "cingul-Ant", "cingul-Mid", "cingul-Post", "precuneus", "Angular",
        "Parahip", "rectus", "subcallosal", "pericallosal", "subparietal",
        "Marginalis", "orbital", "frontomargin", "frontopol", "suborbital",
    ]),
    ("frontoparietal", [
        "front_middle", "front_sup", "parietal_sup", "intrapariet",
        "front_inf", "interm_prim", "precentral-sup", "precentral-inf",
    ]),
]


# Primary sensory Destrieux regions EXCLUDED from the association-cortex mask.
# Neural reliability (the retention metric) lives in higher-order association
# cortex; primary V1/A1 mostly reflect raw stimulus drive (the brainrot-gameable
# part), so we drop them. ("G_cuneus" is early visual; it does NOT match the
# association-cortex "G_precuneus".)
_PRIMARY_SENSORY = [
    "calcarine",          # V1 (calcarine sulcus)
    "Pole_occipital",     # occipital pole (early visual)
    "G_cuneus",           # cuneus (early visual)
    "T_transv",           # Heschl's / transverse temporal gyrus (A1)
    "temporal_transverse",
]


def _classify(region_name: str) -> str:
    for network, keywords in _RULES:
        if any(k in region_name for k in keywords):
            return network
    return "other"


_ATLAS_CACHE = None


def build_atlas():
    """Return (vertex_network: (20484,) str array, mapping: region->network).

    vertex_network[i] is the network name for column i of the prediction array
    (concatenated left then right hemisphere, matching the model's layout).
    """
    global _ATLAS_CACHE
    if _ATLAS_CACHE is not None:
        return _ATLAS_CACHE

    atlas = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas["labels"]]
    region_to_net = {name: _classify(name) for name in labels}
    # Unknown / Medial_wall are not real cortex -> force to "other".
    for junk in ("Unknown", "Medial_wall"):
        if junk in region_to_net:
            region_to_net[junk] = "other"

    map_left = np.asarray(atlas["map_left"]).astype(int)
    map_right = np.asarray(atlas["map_right"]).astype(int)
    label_ids = np.concatenate([map_left, map_right])  # (20484,)
    vertex_network = np.array(
        [region_to_net[labels[i]] for i in label_ids], dtype=object
    )
    _ATLAS_CACHE = (vertex_network, region_to_net, labels, label_ids)
    return _ATLAS_CACHE


def network_masks() -> dict[str, np.ndarray]:
    """Boolean (20484,) mask per named network."""
    vertex_network, _, _, _ = build_atlas()
    return {net: (vertex_network == net) for net in NETWORKS}


def association_cortex_mask() -> np.ndarray:
    """Boolean (20484,) mask: all cortex EXCEPT primary visual/auditory and the
    medial wall. This is the support of the retention / neural-reliability metric
    — higher-order association cortex, DMN included."""
    _, _, labels, label_ids = build_atlas()
    names = np.array([labels[i] for i in label_ids])
    primary = np.array([any(k in n for k in _PRIMARY_SENSORY) for n in names])
    junk = np.isin(names, ["Unknown", "Medial_wall"])
    return ~(primary | junk)


def network_timeseries(preds: np.ndarray, signed: bool = True) -> dict[str, np.ndarray]:
    """Mean activation within each network at every timestep.

    Parameters
    ----------
    preds  : (T, 20484) prediction array.
    signed : if True, average the raw signed z-scored values (shows the
             network's response trajectory, +/-). If False, average |value|
             (magnitude of engagement irrespective of sign).

    Returns
    -------
    {network_name: (T,) array}.
    """
    masks = network_masks()
    x = preds if signed else np.abs(preds)
    return {net: x[:, m].mean(axis=1) for net, m in masks.items()}


def assignment_summary() -> str:
    """Human-readable table of which Destrieux regions feed each network."""
    _, region_to_net, _, _ = build_atlas()
    lines = []
    for net in NETWORKS + ["other"]:
        regs = sorted(r for r, n in region_to_net.items() if n == net)
        lines.append(f"{net} ({len(regs)} regions):")
        lines.append("  " + ", ".join(regs))
    return "\n".join(lines)


def vertex_counts() -> dict[str, int]:
    """Number of vertices (both hemispheres) assigned to each named network."""
    masks = network_masks()
    return {net: int(m.sum()) for net, m in masks.items()}


def plot_network_timeseries(
    preds: np.ndarray,
    times,
    out_path: str = "networks_timeseries.png",
    signed: bool = True,
) -> str:
    """Line plot (5 networks over time) + companion network x time heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = network_timeseries(preds, signed=signed)
    t = np.asarray(times)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]}
    )

    colors = {
        "visual": "#d62728",
        "auditory": "#ff7f0e",
        "language": "#2ca02c",
        "frontoparietal": "#1f77b4",
        "default_mode": "#9467bd",
    }
    for net in NETWORKS:
        ax1.plot(t, ts[net], label=net, color=colors[net], linewidth=2, marker="o", ms=3)
    ax1.axhline(0, color="0.7", lw=0.8)
    ax1.set_ylabel("mean activation\n(z-scored BOLD)" if signed else "mean |activation|")
    ax1.set_title("Per-network predicted brain response over the clip")
    ax1.legend(loc="upper right", fontsize=8, ncol=5)
    ax1.grid(True, alpha=0.3)

    mat = np.vstack([ts[net] for net in NETWORKS])
    vmax = np.percentile(np.abs(mat), 98) or 1.0
    im = ax2.imshow(
        mat, aspect="auto", cmap="coolwarm" if signed else "magma",
        vmin=-vmax if signed else 0, vmax=vmax,
        extent=[t[0], t[-1], len(NETWORKS) - 0.5, -0.5],
    )
    ax2.set_yticks(range(len(NETWORKS)))
    ax2.set_yticklabels(NETWORKS, fontsize=8)
    ax2.set_xlabel("time (s)")
    fig.colorbar(im, ax=ax2, fraction=0.025)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
