"""
Engagement metric: mean|z| over association cortex / mean|z| over primary sensory cortex.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METRIC DEFINITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  engagement(t) =  mean |z| over ASSOCIATION cortex
                   ─────────────────────────────────
                   mean |z| over PRIMARY-SENSORY cortex

  Association cortex (~17,700 vertices): all cortex except primary visual
    (V1/calcarine, occipital pole, cuneus), primary auditory (Heschl's gyrus),
    and the medial wall. Higher-order cortex: meaning, attention, memory.

  Primary sensory cortex (~970 vertices): V1 + A1 — regions that register raw
    light and sound without deeper encoding.

  Why a ratio?
    • Captures cognitive engagement, not passive sensory bombardment.
    • Amplitude-invariant: global gain from video encoding cancels in the ratio.
    • Gaming-resistant: flashy but content-empty video drives the denominator up,
      lowering the score. Engagement only rises when higher-order cortex responds.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HEMODYNAMIC LAG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Brain BOLD response trails the stimulus by ~5 s.
  stimulus_time = brain_time − 5 s  (only score where stimulus_time ≥ 0).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ATLAS: Destrieux 2010 (74 regions × 2 hemispheres on fsaverage5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Left verts 0–10241, right verts 10242–20483. Medial wall = label 0.
"""

import numpy as np

_HEMODYNAMIC_LAG = 5    # seconds
_SMOOTH_WINDOW   = 3    # second moving average

# ── Primary sensory regions (complement → association mask) ────────────────────
#
# V1: calcarine sulcus, occipital pole, cuneus
# A1: Heschl's gyrus (transverse temporal gyrus)

_SENSORY_REGION_SUBSTRINGS = [
    "G_and_S_calcarine",
    "S_calcarine",
    "Pole_occipital",
    "G_cuneus",
    "G_occipital_sup",
    "G_temp_sup-G_T_transv",   # Heschl's gyrus
]


# ── Atlas loading ──────────────────────────────────────────────────────────────

def _load_destrieux():
    from nilearn.datasets import fetch_atlas_surf_destrieux
    d = fetch_atlas_surf_destrieux(mesh="fsaverage5")
    lh = np.asarray(d["map_left"],  dtype=int)
    rh = np.asarray(d["map_right"], dtype=int)
    names = [
        (n.decode() if isinstance(n, bytes) else n)
        for n in d["labels"]
    ]
    return lh, rh, names


def _build_sensory_mask(lh, rh, names):
    vertices = []
    for substring in _SENSORY_REGION_SUBSTRINGS:
        matching = [i for i, n in enumerate(names) if substring in n]
        for label_idx in matching:
            vertices.extend(np.where(lh == label_idx)[0].tolist())
            vertices.extend((np.where(rh == label_idx)[0] + 10242).tolist())
    return np.unique(np.array(vertices, dtype=int))


def _approximate_masks() -> dict[str, np.ndarray]:
    """Fallback when nilearn is not installed."""
    lh, rh = 0, 10242
    sensory = np.concatenate([
        np.arange(lh + 700,  lh + 1100),
        np.arange(lh + 3200, lh + 3450),
        np.arange(rh + 700,  rh + 1100),
        np.arange(rh + 3200, rh + 3450),
    ])
    medial = np.concatenate([
        np.arange(lh + 9800, lh + 10242),
        np.arange(rh + 9800, rh + 10242),
    ])
    association = np.setdiff1d(np.arange(20484), np.union1d(sensory, medial))
    return {"association_cortex": association, "primary_sensory": sensory}


_MASKS: dict[str, np.ndarray] | None = None


def get_masks() -> dict[str, np.ndarray]:
    global _MASKS
    if _MASKS is not None:
        return _MASKS
    try:
        lh, rh, names = _load_destrieux()
        sensory   = _build_sensory_mask(lh, rh, names)
        medial_lh = np.where(lh == 0)[0]
        medial_rh = np.where(rh == 0)[0] + 10242
        exclude   = np.union1d(sensory, np.concatenate([medial_lh, medial_rh]))
        assoc     = np.setdiff1d(np.arange(20484), exclude)
        _MASKS    = {"association_cortex": assoc, "primary_sensory": sensory}
    except Exception:
        _MASKS = _approximate_masks()
    return _MASKS


# ── Scoring ────────────────────────────────────────────────────────────────────

def _moving_avg(arr: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(arr, np.ones(window) / window, mode="same")


def score_preds(preds: np.ndarray) -> dict:
    """
    Score a (T, 20484) TRIBE prediction array.

    Applies 5-second hemodynamic lag correction, computes per-second engagement
    ratio mean|z|(association) / mean|z|(sensory), then 3-second moving average.

    Returns:
      association_cortex   — smoothed mean |z| over valid window
      primary_sensory      — smoothed mean |z| over valid window
      reward               — mean engagement ratio (the optimization target)
      peak_sensory_second  — stimulus second where sensory dominates most
      peak_memory_second   — stimulus second with highest engagement ratio
      per_second           — per-second list (indexed by stimulus time)
    """
    masks = get_masks()
    lag   = _HEMODYNAMIC_LAG

    preds_valid = preds[lag:]          # brain_time[lag:] = stimulus_time[0:]
    T_valid     = preds_valid.shape[0]

    assoc_ts   = np.abs(preds_valid[:, masks["association_cortex"]]).mean(axis=1)
    sensory_ts = np.abs(preds_valid[:, masks["primary_sensory"]]).mean(axis=1)
    rew_ts     = assoc_ts / np.maximum(sensory_ts, 1e-8)

    assoc_s   = _moving_avg(assoc_ts,   _SMOOTH_WINDOW)
    sensory_s = _moving_avg(sensory_ts, _SMOOTH_WINDOW)
    rew_s     = _moving_avg(rew_ts,     _SMOOTH_WINDOW)

    per_second = [
        {
            "second":             int(t),
            "association_cortex": float(assoc_s[t]),
            "primary_sensory":    float(sensory_s[t]),
            "reward":             float(rew_s[t]),
        }
        for t in range(T_valid)
    ]

    return {
        "association_cortex":  float(assoc_s.mean()),
        "primary_sensory":     float(sensory_s.mean()),
        "reward":              float(rew_s.mean()),
        "peak_sensory_second": int(sensory_s.argmax()),
        "peak_memory_second":  int(rew_s.argmax()),
        "per_second":          per_second,
    }
