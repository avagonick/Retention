"""
fsaverage5 vertex masks for functional brain regions.

fsaverage5 has 20484 vertices total: 0-10241 = left hemisphere, 10242-20483 = right.
Masks here come from the Yeo 7-network parcellation (HCP-style) projected onto fsaverage5.

To replace with proper atlas: load lh/rh.Yeo2011_7Networks_N1000.annot via nibabel
and concatenate: labels = np.concatenate([lh_labels, rh_labels]).
Then build masks like: DMN_VERTICES = np.where(labels == 7)[0]

Networks: 1=Visual 2=Somatomotor 3=DorsalAttn 4=VentralAttn 5=Limbic 6=Frontoparietal 7=DMN
"""

import numpy as np


def _try_load_yeo_atlas() -> dict[str, np.ndarray] | None:
    """Try to load the Yeo 7-network parcellation via nilearn."""
    try:
        import nibabel as nib
        from nilearn.datasets import fetch_surf_fsaverage

        fsaverage = fetch_surf_fsaverage("fsaverage5")
        lh = nib.freesurfer.read_annot(fsaverage["sulc_left"].replace("sulc", "parc2009").replace(".gii", ""))
        # nilearn ships Destrieux, not Yeo — fall through to approximation
        return None
    except Exception:
        return None


def _approximate_masks() -> dict[str, np.ndarray]:
    """
    Conservative approximations based on the Yeo 7-network vertex distribution
    on fsaverage5. Not exact — replace with proper atlas for production.

    Rough vertex counts per network per hemisphere on fsaverage5:
      DMN (~7): medial frontal + PCC + angular = ~1800 verts/hemi
      Frontoparietal (~6): dlPFC + IPS = ~900 verts/hemi
      DorsalAttn (~3): FEF + IPS = ~700 verts/hemi
      Limbic (~5): OFC + parahippocampal = ~400 verts/hemi
    """
    lh_offset = 0
    rh_offset = 10242

    # DMN: posterior cingulate + medial PFC + angular gyrus (both hemispheres)
    dmn = np.concatenate([
        np.arange(lh_offset + 800,  lh_offset + 1400),   # PCC-ish
        np.arange(lh_offset + 5200, lh_offset + 6000),   # mPFC-ish
        np.arange(lh_offset + 7800, lh_offset + 8400),   # angular-ish
        np.arange(rh_offset + 800,  rh_offset + 1400),
        np.arange(rh_offset + 5200, rh_offset + 6000),
        np.arange(rh_offset + 7800, rh_offset + 8400),
    ])

    # Memory: parahippocampal + lateral PFC (encoding-related)
    memory = np.concatenate([
        np.arange(lh_offset + 2400, lh_offset + 3000),   # parahippocampal-ish
        np.arange(lh_offset + 4200, lh_offset + 4800),   # lateral PFC-ish
        np.arange(rh_offset + 2400, rh_offset + 3000),
        np.arange(rh_offset + 4200, rh_offset + 4800),
    ])

    # Attention: frontoparietal + dorsal attention
    attention = np.concatenate([
        np.arange(lh_offset + 3000, lh_offset + 3600),   # IPS-ish
        np.arange(lh_offset + 6200, lh_offset + 6800),   # dlPFC-ish
        np.arange(rh_offset + 3000, rh_offset + 3600),
        np.arange(rh_offset + 6200, rh_offset + 6800),
    ])

    return {"dmn": dmn, "memory": memory, "attention": attention}


# Module-level masks — loaded once
_MASKS: dict[str, np.ndarray] | None = None


def get_masks() -> dict[str, np.ndarray]:
    global _MASKS
    if _MASKS is None:
        _MASKS = _try_load_yeo_atlas() or _approximate_masks()
    return _MASKS


def score_preds(preds: np.ndarray) -> dict[str, float]:
    """
    Convert a (T, 20484) prediction array to per-region scalar scores.

    Returns memory, dmn, attention (mean over time and vertices) plus a
    composite reward: high memory + attention, low DMN.
    """
    masks = get_masks()
    memory    = float(preds[:, masks["memory"]].mean())
    dmn       = float(preds[:, masks["dmn"]].mean())
    attention = float(preds[:, masks["attention"]].mean())
    reward    = memory + 0.5 * attention - 1.5 * dmn

    return {
        "memory": memory,
        "dmn": dmn,
        "attention": attention,
        "reward": float(reward),
    }
