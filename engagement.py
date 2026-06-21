"""Engagement score — the fitness function the optimization loop maximizes.

Definition (chosen design):

    sensory(t)    = mean(visual, auditory, language)          # stimulus drive
    engagement(t) = w_att * frontoparietal(t)                 # task-positive
                  + w_sens * sensory(t)                        # sensory/linguistic pull
                  - w_dmn * default_mode(t)                    # task-negative (mind-wandering)

Higher = more "focused engagement". Weights are exposed so the loop can tune it.

Two *separate* uses of this curve, deliberately decoupled:

  1. GLOBAL FITNESS — the single number compared across loop iterations. It is a
     mean over the whole curve, so it is shift-invariant: the ~5 s hemodynamic
     lag does NOT require shifting or truncating anything here. Every row counts.

  2. EDIT LOCALIZATION — which video second to fix. This is the ONLY place the
     lag matters: a low-engagement BOLD row at brain-time t was driven by the
     stimulus ~lag_s earlier, so the edit target is `t - lag_s`. The lag is an
     offset applied at lookup time, not a reason to drop data.

Edge reality: BOLD for the final ~lag_s seconds of content lands in rows that
would occur AFTER the clip ends, which TRIBE never returns. So edits can only be
localized up to `(clip_end - lag_s)`. To recover the very end, tail-pad the clip
(~lag_s+1 s of freeze-frame/black) before scoring — see `video_utils.pad_video_tail`
— then the response to the real ending falls in real returned rows.

Time smoothing: a short centred moving average (default 3 s) denoises the
sluggish BOLD without distorting the slow dynamics.
"""

from __future__ import annotations

import numpy as np

from brain_networks import NETWORKS, network_timeseries

# Default weights for the Attention - DMN construct.
DEFAULT_WEIGHTS = {"attention": 1.0, "sensory": 1.0, "dmn": 1.0}

DEFAULT_LAG_S = 5.0
DEFAULT_SMOOTH_S = 3


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge replication (length-preserving)."""
    if window <= 1:
        return x
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def compute_engagement(
    preds: np.ndarray,
    times,
    weights: dict | None = None,
    lag_s: float = DEFAULT_LAG_S,
    smooth_window: int = DEFAULT_SMOOTH_S,
) -> dict:
    """Compute the engagement curve, the global fitness scalar, and edit targets.

    Returns a dict with:
      networks        : {name: (T,) smoothed series}
      attention/sensory/default_mode : (T,) component series (smoothed)
      engagement      : (T,) engagement per BOLD row (brain-time) — nothing dropped
      times_brain     : (T,) TRIBE row times (brain-time)
      times_stimulus  : (T,) brain-time - lag_s (the video moment that drove each row)
      overall         : float — mean engagement over ALL rows == the fitness value
      targetable      : (T,) bool — times_stimulus >= 0 (edit cause lies inside the clip)
      coverage_end    : float — last stimulus second that can be localized (clip_end - lag_s)
      weakest         : {t_stimulus, t_brain, engagement} — lowest targetable moment (edit here)
      weights, lag_s, smooth_window
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    raw = network_timeseries(preds, signed=True)
    nets = {name: _smooth(raw[name], smooth_window) for name in NETWORKS}

    attention = nets["frontoparietal"]
    sensory = np.mean([nets["visual"], nets["auditory"], nets["language"]], axis=0)
    dmn = nets["default_mode"]

    engagement = (
        w["attention"] * attention
        + w["sensory"] * sensory
        - w["dmn"] * dmn
    )

    times_brain = np.asarray(times, dtype=float)
    times_stimulus = times_brain - lag_s
    targetable = times_stimulus >= 0

    # 1) Global fitness: mean over the whole curve. Shift-invariant, lossless.
    overall = float(engagement.mean())

    # 2) Edit localization: weakest moment whose cause lies inside the clip.
    if targetable.any():
        idx = np.where(targetable)[0]
        wi = idx[int(np.argmin(engagement[idx]))]
    else:  # clip shorter than the lag — fall back to the global minimum
        wi = int(np.argmin(engagement))
    weakest = {
        "t_stimulus": float(times_stimulus[wi]),
        "t_brain": float(times_brain[wi]),
        "engagement": float(engagement[wi]),
    }

    return {
        "networks": nets,
        "attention": attention,
        "sensory": sensory,
        "default_mode": dmn,
        "engagement": engagement,
        "times_brain": times_brain,
        "times_stimulus": times_stimulus,
        "overall": overall,
        "targetable": targetable,
        "coverage_end": float(times_brain.max() - lag_s),
        "weakest": weakest,
        "weights": w,
        "lag_s": lag_s,
        "smooth_window": smooth_window,
    }


def to_payload(result: dict) -> dict:
    """JSON-serializable summary for the LLM edit-proposer step.

    The global fitness uses every row; per-row entries carry both brain-time and
    the lag-corrected stimulus time, plus a `targetable` flag so the LLM only
    proposes edits where the cause lies inside the clip.
    """
    tb = result["times_brain"]
    ts = result["times_stimulus"]
    eng = result["engagement"]
    series = [
        {
            "t_brain": round(float(tb[i]), 1),
            "t_stimulus": round(float(ts[i]), 1),
            "targetable": bool(result["targetable"][i]),
            "engagement": round(float(eng[i]), 4),
            "attention": round(float(result["attention"][i]), 4),
            "sensory": round(float(result["sensory"][i]), 4),
            "default_mode": round(float(result["default_mode"][i]), 4),
        }
        for i in range(len(tb))
    ]
    return {
        "overall_engagement": round(result["overall"], 4),
        "definition": "engagement = attention + mean(visual,auditory,language) - default_mode",
        "weights": result["weights"],
        "lag_s": result["lag_s"],
        "edit_target": {
            "t_stimulus": round(result["weakest"]["t_stimulus"], 1),
            "engagement": round(result["weakest"]["engagement"], 4),
        },
        "coverage_end_s": round(result["coverage_end"], 1),
        "note": (
            "Localize edits by t_stimulus. Content after coverage_end_s is not "
            "directly localizable unless the clip was tail-padded before scoring."
        ),
        "series": series,
    }


def plot_engagement(result: dict, out_path: str = "engagement.png") -> str:
    """Engagement over brain-time (lossless) with a lag-corrected stimulus axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tb = result["times_brain"]
    lag = result["lag_s"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(tb, result["engagement"], color="black", lw=2.5, marker="o", ms=4, label="engagement")
    ax1.axhline(0, color="0.7", lw=0.8)
    # mark the weakest targetable moment (the edit target), in brain-time
    ax1.axvline(result["weakest"]["t_brain"], color="red", ls="--", lw=1.2,
                label=f"edit ≈ stimulus {result['weakest']['t_stimulus']:.0f}s")
    # shade the tail that can't be localized without padding (stimulus > coverage_end)
    cov_brain = result["coverage_end"] + lag
    if cov_brain < tb.max():
        ax1.axvspan(cov_brain, tb.max(), color="#ffe8e8", label="tail needs padding")
    ax1.set_ylabel("engagement\n(attention + sensory − DMN)")
    ax1.set_title(
        f"Engagement   ·   overall fitness = {result['overall']:+.3f}   "
        f"(lossless mean; lag −{lag:.0f}s shown on top axis, smooth {result['smooth_window']}s)"
    )
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    # secondary top axis: stimulus time = brain - lag
    secx = ax1.secondary_xaxis("top", functions=(lambda x: x - lag, lambda x: x + lag))
    secx.set_xlabel("stimulus time (s)  =  brain-time − lag")

    ax2.plot(tb, result["attention"], label="+ attention (frontoparietal)", color="#1f77b4", lw=1.8)
    ax2.plot(tb, result["sensory"], label="+ sensory (vis+aud+lang)", color="#2ca02c", lw=1.8)
    ax2.plot(tb, -result["default_mode"], label="− default-mode", color="#9467bd", lw=1.8)
    ax2.axhline(0, color="0.7", lw=0.8)
    ax2.set_ylabel("component contribution")
    ax2.set_xlabel("brain-time (s)  =  TRIBE row time")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
