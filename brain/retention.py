"""Neural-reliability ("retention") metric â€” the recommended fitness function.

Why this over the network-weighted engagement score: for *naturalistic video*,
inter-subject correlation / neural reliability is the validated predictor of both
memory and engagement (Dmochowski 2014; Hasson; Simony 2016), and it lives in
higher-order association cortex, not primary sensory. TRIBE predicts the
AVERAGE-subject response â€” which is itself the shared, stimulus-locked
(reliable) component â€” so we read the *magnitude* of that prediction over
association cortex.

    retention(t) = mean( |predicted z| ) over higher-order association cortex
                   (all cortex minus primary visual/auditory and medial wall;
                   DMN INCLUDED â€” it supports narrative memory, so we do not
                   subtract it the way the engagement metric does).

Magnitude (|.|) because reliability is about how strongly a region is driven,
regardless of sign. Same lag/aggregation conventions as engagement.py: the
global fitness is the lossless mean over all rows; the ~5 s lag is applied only
when localizing an edit.
"""

from __future__ import annotations

import numpy as np

from brain.networks import association_cortex_mask

DEFAULT_LAG_S = 5.0
DEFAULT_SMOOTH_S = 3


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def compute_retention(
    preds: np.ndarray,
    times,
    lag_s: float = DEFAULT_LAG_S,
    smooth_window: int = DEFAULT_SMOOTH_S,
) -> dict:
    """Neural-reliability curve, global fitness, and the weakest targetable moment."""
    mask = association_cortex_mask()
    raw = np.abs(preds[:, mask]).mean(axis=1)  # mean |z| over association cortex
    series = _smooth(raw, smooth_window)

    times_brain = np.asarray(times, dtype=float)
    times_stimulus = times_brain - lag_s
    targetable = times_stimulus >= 0

    # Fitness = mean over the CONTENT window only (stimulus >= 0). Excluding the
    # pre-stimulus hemodynamic ramp keeps a length-dependent baseline out of the
    # score (the ramp is a fixed ~lag seconds, so it dilutes long clips less than
    # short ones â€” a confound for a loop that changes video length).
    overall = float(series[targetable].mean()) if targetable.any() else float(series.mean())

    if targetable.any():
        idx = np.where(targetable)[0]
        wi = idx[int(np.argmin(series[idx]))]
    else:
        wi = int(np.argmin(series))
    weakest = {
        "t_stimulus": float(times_stimulus[wi]),
        "t_brain": float(times_brain[wi]),
        "retention": float(series[wi]),
    }

    return {
        "retention": series,
        "times_brain": times_brain,
        "times_stimulus": times_stimulus,
        "overall": overall,
        "targetable": targetable,
        "coverage_end": float(times_brain.max() - lag_s),
        "weakest": weakest,
        "n_vertices": int(mask.sum()),
        "lag_s": lag_s,
        "smooth_window": smooth_window,
    }


def to_payload(result: dict) -> dict:
    """JSON-serializable summary for the LLM edit-proposer step."""
    tb, ts, r = result["times_brain"], result["times_stimulus"], result["retention"]
    return {
        "overall_retention": round(result["overall"], 4),
        "definition": "mean(|predicted z|) over higher-order association cortex (neural reliability)",
        "lag_s": result["lag_s"],
        "n_vertices": result["n_vertices"],
        "edit_target": {
            "t_stimulus": round(result["weakest"]["t_stimulus"], 1),
            "retention": round(result["weakest"]["retention"], 4),
        },
        "coverage_end_s": round(result["coverage_end"], 1),
        "series": [
            {
                "t_brain": round(float(tb[i]), 1),
                "t_stimulus": round(float(ts[i]), 1),
                "targetable": bool(result["targetable"][i]),
                "retention": round(float(r[i]), 4),
            }
            for i in range(len(tb))
        ],
    }


def plot_retention(result: dict, out_path: str = "retention.png") -> str:
    """Retention curve over brain-time, with a lag-corrected stimulus axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tb = result["times_brain"]
    lag = result["lag_s"]
    fig, ax = plt.subplots(figsize=(11, 4.6))

    ax.plot(tb, result["retention"], color="#1f77b4", lw=2.5, marker="o", ms=4,
            label="retention (neural reliability)")
    ax.axvline(result["weakest"]["t_brain"], color="red", ls="--", lw=1.2,
               label=f"edit â‰ˆ stimulus {result['weakest']['t_stimulus']:.0f}s")
    # shade the pre-stimulus warm-up (brain-time < lag, stimulus < 0) â€” excluded from fitness
    if float(tb.min()) < lag:
        ax.axvspan(tb.min(), lag, color="0.88", label="pre-stimulus ramp (excluded)")
    cov_brain = result["coverage_end"] + lag
    if cov_brain < tb.max():
        ax.axvspan(cov_brain, tb.max(), color="#ffe8e8", label="tail needs padding")

    ax.set_ylabel("retention\n(mean |z| over association cortex)")
    ax.set_xlabel("brain-time (s)  =  TRIBE row time")
    ax.set_title(
        f"Retention / neural reliability   Â·   overall fitness = {result['overall']:.3f}   "
        f"(mean over stimulusâ‰¥0; lag âˆ’{lag:.0f}s on top axis; {result['n_vertices']} verts)"
    )
    secx = ax.secondary_xaxis("top", functions=(lambda x: x - lag, lambda x: x + lag))
    secx.set_xlabel("stimulus time (s)  =  brain-time âˆ’ lag")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
