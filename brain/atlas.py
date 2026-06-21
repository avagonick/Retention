"""
Maps the four target regions to fsaverage5 vertex indices using the Destrieux atlas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY CORTICAL PROXIES FOR HIPPOCAMPUS AND AMYGDALA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fsaverage5 is a cortical surface mesh (20,484 vertices). Hippocampus and amygdala
are subcortical — they have no surface vertices. TRIBE predicts cortical BOLD.
We use the closest cortical regions anatomically and functionally:

  Hippocampus → parahippocampal cortex
      The parahippocampal gyrus (PHC) is the direct cortical output of the
      hippocampus. Entorhinal → PHC is the canonical memory encoding pathway.
      PHC BOLD tracks hippocampal encoding success (Davachi 2006, Staresina 2012).

  Amygdala → temporal pole (anterior inferior temporal)
      The temporal pole receives dense amygdala projections and codes the
      emotional valence of stimuli. Temporal pole BOLD correlates with
      amygdala responses to affectively significant content (Olson 2007).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ATLAS: Destrieux 2010 (74 regions × 2 hemispheres on fsaverage5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loaded via nilearn.datasets.fetch_atlas_surf_destrieux(mesh='fsaverage5').
Returns per-vertex integer labels for left (10242 verts) and right (10242 verts).
Full surface: left verts 0–10241, right verts 10242–20483.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORE FORMULA (per second t)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  reward(t) = hippocampus(t) + left_pfc(t) + amygdala(t) − 2.0 × dmn(t)

  Each term = mean TRIBE activation over that region's vertices at second t.
  DMN weighted 2× because active mind-wandering is a stronger negative signal
  than any single encoding region failing to activate.
"""

import numpy as np

# ── Destrieux region name substrings → target anatomy ─────────────────────────
#
# Matching is substring-based so minor nilearn version differences in label text
# don't break the lookup. Each entry is (substring, hemisphere).

_HIPPOCAMPUS_REGIONS = [
    # Parahippocampal gyrus — direct cortical relay of hippocampal output
    ("Parahip", "both"),
    # Entorhinal cortex — perforant path origin, feeds into hippocampus
    ("G_oc-temp_med-Lingual", "both"),   # lingual gyrus borders PHC
]

_LEFT_PFC_REGIONS = [
    # Broca's area (semantic processing depth)
    ("G_front_inf-Opercular", "left"),
    ("G_front_inf-Triangul",  "left"),
    ("G_front_inf-Orbital",   "left"),
    # Dorsolateral PFC (working memory, elaborative encoding)
    ("G_front_middle",        "left"),
    ("G_front_sup",           "left"),
]

_AMYGDALA_REGIONS = [
    # Temporal pole — densest amygdala cortical projection target
    ("G_temporal_inf",        "both"),
    ("G_temp_sup-Plan_polar", "both"),
]

_DMN_REGIONS = [
    # Posterior cingulate cortex — primary DMN hub (Buckner 2008)
    ("G_cingul-Post-dorsal",  "both"),
    ("G_cingul-Post-ventral", "both"),
    # Precuneus — episodic retrieval / self-referential processing
    ("G_precuneus",           "both"),
    # Medial PFC — self-referential thought
    ("G_and_S_cingul-Ant",   "both"),
    ("G_rectus",              "both"),
    # Angular gyrus — semantic memory / mind-wandering
    ("G_pariet_inf-Angular",  "both"),
]


# ── Atlas loading ──────────────────────────────────────────────────────────────

def _load_destrieux():
    """Return (lh_labels, rh_labels, label_names) for fsaverage5."""
    from nilearn.datasets import fetch_atlas_surf_destrieux
    d = fetch_atlas_surf_destrieux(mesh="fsaverage5")
    lh = np.asarray(d["map_left"],  dtype=int)   # (10242,)
    rh = np.asarray(d["map_right"], dtype=int)   # (10242,)
    names = [
        (n.decode() if isinstance(n, bytes) else n)
        for n in d["labels"]
    ]
    return lh, rh, names


def _build_mask(region_specs: list[tuple], lh: np.ndarray, rh: np.ndarray, names: list[str]) -> np.ndarray:
    """
    For each (substring, hemisphere) spec, find matching Destrieux label indices,
    collect the surface vertices for that hemisphere, return concatenated array.
    """
    vertices: list[int] = []
    for substring, hemi in region_specs:
        matching = [i for i, n in enumerate(names) if substring in n]
        if not matching:
            continue
        for label_idx in matching:
            if hemi in ("left", "both"):
                vertices.extend(np.where(lh == label_idx)[0].tolist())
            if hemi in ("right", "both"):
                vertices.extend((np.where(rh == label_idx)[0] + 10242).tolist())
    return np.unique(np.array(vertices, dtype=int))


def _approximate_masks() -> dict[str, np.ndarray]:
    """Fallback when nilearn is not installed — coarse anatomical approximations."""
    lh, rh = 0, 10242
    return {
        "hippocampus": np.concatenate([np.arange(lh+2400, lh+3000), np.arange(rh+2400, rh+3000)]),
        "left_pfc":    np.arange(lh+4200, lh+5200),
        "amygdala":    np.concatenate([np.arange(lh+1800, lh+2400), np.arange(rh+1800, rh+2400)]),
        "dmn":         np.concatenate([np.arange(lh+800, lh+1400), np.arange(lh+5200, lh+6000),
                                        np.arange(rh+800, rh+1400), np.arange(rh+5200, rh+6000)]),
    }


_MASKS: dict[str, np.ndarray] | None = None


def get_masks() -> dict[str, np.ndarray]:
    global _MASKS
    if _MASKS is not None:
        return _MASKS
    try:
        lh, rh, names = _load_destrieux()
        _MASKS = {
            "hippocampus": _build_mask(_HIPPOCAMPUS_REGIONS, lh, rh, names),
            "left_pfc":    _build_mask(_LEFT_PFC_REGIONS,    lh, rh, names),
            "amygdala":    _build_mask(_AMYGDALA_REGIONS,    lh, rh, names),
            "dmn":         _build_mask(_DMN_REGIONS,         lh, rh, names),
        }
    except Exception:
        _MASKS = _approximate_masks()
    return _MASKS


# ── Scoring ────────────────────────────────────────────────────────────────────

_DMN_WEIGHT = 2.0


def score_preds(preds: np.ndarray) -> dict:
    """
    Score a (T, 20484) prediction array against the four target regions.

    Per-second scores let the generator see WHEN the viewer's mind wandered
    or WHEN memory encoding spiked — not just overall averages.

    Returns:
      hippocampus, left_pfc, amygdala, dmn  — mean over time
      reward                                 — composite score
      peak_dmn_second                        — worst mind-wandering moment
      peak_memory_second                     — best encoding moment
      per_second                             — list of per-second dicts
    """
    masks = get_masks()
    T = preds.shape[0]

    hipp_ts = preds[:, masks["hippocampus"]].mean(axis=1)
    pfc_ts  = preds[:, masks["left_pfc"]].mean(axis=1)
    amyg_ts = preds[:, masks["amygdala"]].mean(axis=1)
    dmn_ts  = preds[:, masks["dmn"]].mean(axis=1)
    rew_ts  = hipp_ts + pfc_ts + amyg_ts - _DMN_WEIGHT * dmn_ts

    per_second = [
        {
            "second":      int(t),
            "hippocampus": float(hipp_ts[t]),
            "left_pfc":    float(pfc_ts[t]),
            "amygdala":    float(amyg_ts[t]),
            "dmn":         float(dmn_ts[t]),
            "reward":      float(rew_ts[t]),
        }
        for t in range(T)
    ]

    return {
        "hippocampus":       float(hipp_ts.mean()),
        "left_pfc":          float(pfc_ts.mean()),
        "amygdala":          float(amyg_ts.mean()),
        "dmn":               float(dmn_ts.mean()),
        "reward":            float(rew_ts.mean()),
        "peak_dmn_second":    int(dmn_ts.argmax()),
        "peak_memory_second": int(rew_ts.argmax()),
        "per_second":        per_second,
    }
