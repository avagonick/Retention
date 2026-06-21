"""
Evaluator agent — brain scoring + vision panel feedback for every iteration.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THE DISCRIMINATOR DOES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Runs for every iteration. Always. No early stopping — we're doing
best-of-N and the evaluator is the scorer, not the judge.

Per iteration:

  ┌─────────────────────────────────────────────────────────────┐
  │  1. TRIBE brain scoring  (objective signal)                 │
  │                                                             │
  │     video → TRIBE v2 on Lightning AI                        │
  │           → (T, 20484) cortical activation per second       │
  │           → mean activation over 4 region vertex masks:     │
  │               hippocampus  (parahippocampal cortex)         │
  │               left_pfc     (IFG + MFG)                      │
  │               amygdala     (temporal pole)                  │
  │               dmn          (PCC + precuneus + mPFC)         │
  │           → reward = hipp + pfc + amyg − 2×dmn             │
  │                                                             │
  │     Best-of-5 tracking: highest reward wins.               │
  ├─────────────────────────────────────────────────────────────┤
  │  2. Vision panel feedback  (interpretation layer)           │
  │                                                             │
  │     3 models in parallel via TokenRouter:                   │
  │       claude-opus-4-8 / gpt-4o / claude-3-5-sonnet          │
  │                                                             │
  │     Each judge sees:                                        │
  │       • question being taught                               │
  │       • per-second brain score table                        │
  │       • frame at peak_dmn_second (where viewer zoned out)   │
  │       • frame at peak_memory_second (best encoding moment)  │
  │       • generator params used this iteration                │
  │       • history of prior iteration scores + feedback        │
  │                                                             │
  │     Judges answer: WHY did the brain respond this way?      │
  │     What specifically should the generator change?          │
  │     → structured feedback JSON (no verdict — no voting)     │
  │                                                             │
  │     Feedbacks from all 3 judges are merged.                 │
  ├─────────────────────────────────────────────────────────────┤
  │  3. Feedback sent to generator                              │
  │                                                             │
  │     Merged feedback goes to generator so it can reason      │
  │     over brain data + visual context for the next attempt.  │
  └─────────────────────────────────────────────────────────────┘

WHY LLM + BRAIN SCORES TOGETHER?

  Brain scores are objective but uninterpretable by a video generator.
  "DMN=0.44 at t=8s" doesn't tell Pika what to change.

  The LLM bridges the gap: it sees the actual frame at t=8s,
  sees DMN=0.44 there, and reasons: "the screen shows 6 bullet
  points — too much to parse, simplify to one concept."

Set env vars:
  TRIBE_BASE_URL        — Lightning AI base URL (TRIBE_ENDPOINT also accepted)
  TRIBE_API_TOKEN       — X-API-Token for the TRIBE endpoint
  TOKEN_ROUTER_API_KEY  — TokenRouter sk-... key
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import httpx
import numpy as np
from openai import AsyncOpenAI

from brain.atlas import score_preds
from brain.frames import extract_all_frames_base64
from brain.tribe_client import score as tribe_score
from .band import Band

logger = logging.getLogger(__name__)

# Score via the robust client in brain/tribe_client.py. Accept either env name;
# strip a trailing /score so the client can append its own /score and /health.
TRIBE_BASE_URL = (os.getenv("TRIBE_BASE_URL") or os.getenv("TRIBE_ENDPOINT", "")).removesuffix("/score").rstrip("/")
TRIBE_API_TOKEN = os.getenv("TRIBE_API_TOKEN", "")

_TOKEN_ROUTER_URL = "https://api.tokenrouter.com/v1"
_PANEL_MODELS = [
    "claude-opus-4-8",  # deep reasoning, strong at structured output
    "gpt-4o",           # strong visual grounding, independent perspective
    "gemini-1.5-pro",   # different training distribution, independent read
]

_SYSTEM = """\
You are evaluating an AI-generated educational video for cognitive engagement quality.
You have access to fMRI brain activation data AND visual frames from the video.

METRIC: engagement = mean|z|(association cortex) / mean|z|(primary sensory cortex)

  Association cortex (~17,700 vertices): all higher-order cortex — meaning, memory, attention.
  Primary sensory cortex (~970 vertices): V1 + A1 — raw visual/audio perception only.

  Higher engagement = brain is actively understanding, not just passively watching.
  The score is amplitude-invariant (ratio cancels global gain) and gaming-resistant
  (flashy, empty content drives sensory cortex up, which lowers the ratio).

You will see:
  1. The per-second engagement table (association, sensory, ratio per second)
  2. A frame from the WORST second (peak sensory — viewer is perceiving but not encoding)
  3. A frame from the BEST second (peak engagement — deepest meaning-making)

Use the frames to understand WHY the brain shifted between passive perception and
active understanding. Reference what you see in the frames in your explanation.

Respond with valid JSON only — no markdown:
{
  "reason": "<one sentence — cite what you see in the frame AND the engagement score>",
  "feedback": {
    "pacing": "faster" | "slower" | "ok",
    "visual_complexity": "increase" | "decrease" | "ok",
    "add_text_overlays": true | false,
    "add_motion": true | false,
    "style_note": "<concrete visual instruction referencing what you saw in the frames>"
  }
}\
"""


# ── TRIBE inference ────────────────────────────────────────────────────────────

async def _call_tribe(video_path: str) -> np.ndarray:
    # brain.tribe_client.score is sync (requests) — run it off the event loop.
    preds, _times = await asyncio.to_thread(
        tribe_score, video_path, TRIBE_BASE_URL, TRIBE_API_TOKEN
    )
    return preds


# ── LLM vision panel via TokenRouter ──────────────────────────────────────────

def _build_text_prompt(
    question: str,
    scores: dict,
    params: dict,
    iteration: int,
    history: list[dict],
) -> str:
    prior = ""
    if history:
        prior = "\nPrior iterations:\n" + "\n".join(
            f"  iter {i+1}: reward={h['scores']['reward']:.3f} — {h['reason']}"
            for i, h in enumerate(history)
        ) + "\n"

    return (
        f'Question being taught: "{question}"\n'
        f"Iteration: {iteration} of 5\n"
        f"Generator params: {json.dumps(params)}\n"
        f"{prior}\n"
        f"Overall means:  "
        f"association={scores['association_cortex']:.3f}  "
        f"sensory={scores['primary_sensory']:.3f}  "
        f"engagement={scores['reward']:.3f}\n"
        f"Worst second (peak sensory): t={scores['peak_sensory_second']}s  |  "
        f"Best second (peak engagement): t={scores['peak_memory_second']}s\n\n"
        "Below: each second's engagement score followed by the video frame at that second.\n"
        "Use this to explain exactly WHY the brain shifted between passive perception and active understanding."
    )


def _build_vision_messages(
    text_prompt: str,
    all_frames: list[tuple[int, str]],
    per_second: list[dict],
) -> list[dict]:
    """
    Build the message with every second's frame and brain score interleaved.
    The model sees: score line → frame → score line → frame → ...
    so it can directly connect what was on screen to how the brain responded.
    """
    # Index scores by second for O(1) lookup
    score_by_second = {r["second"]: r for r in per_second}

    content: list[dict] = [{"type": "text", "text": text_prompt}]

    for second, b64 in all_frames:
        s = score_by_second.get(second, {})
        label = (
            f"t={second}s — "
            f"engagement={s.get('reward', '?'):.3f}  "
            f"assoc={s.get('association_cortex', '?'):.3f}  "
            f"sensory={s.get('primary_sensory', '?'):.3f}"
        )
        content.append({"type": "text", "text": label})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    content.append({"type": "text", "text": "Provide your feedback as JSON."})
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": content}]


async def _call_single_judge(model: str, client: AsyncOpenAI, messages: list[dict]) -> dict:
    response = await client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=messages,
    )
    result = json.loads(response.choices[0].message.content.strip())
    result["model"] = model
    return result


_SYNTHESIS_SYSTEM = """\
You are synthesizing feedback from three independent AI evaluators of an educational video.
Each evaluator saw the same fMRI brain scores and video frames and gave feedback.

Your job: produce one coherent, non-contradictory feedback instruction for the video generator.

Rules:
- If evaluators disagree on pacing/visual_complexity, choose based on which the brain data supports.
  High DMN + low reward → the viewer is zoned out → usually means pacing too slow or complexity too high.
- For add_text_overlays / add_motion: recommend if 2+ evaluators agree OR if the brain data strongly suggests it.
- Write style_note as a single concrete instruction (not a concatenation of three opinions).
  Pick the most specific and actionable observation across all three.
- reason: one sentence explaining the dominant finding across evaluators.

Respond with valid JSON only — no markdown:
{
  "reason": "<dominant finding across all three evaluators>",
  "feedback": {
    "pacing": "faster" | "slower" | "ok",
    "visual_complexity": "increase" | "decrease" | "ok",
    "add_text_overlays": true | false,
    "add_motion": true | false,
    "style_note": "<single concrete instruction for the generator>"
  }
}\
"""


def _make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=_TOKEN_ROUTER_URL,
        api_key=os.environ["TOKEN_ROUTER_API_KEY"],
    )


def _merge_feedback(feedbacks: list[dict]) -> dict:
    """Fallback merge used only when synthesis call fails."""
    return {
        "pacing": next(
            (f["pacing"] for f in feedbacks if f.get("pacing") != "ok"), "ok"
        ),
        "visual_complexity": next(
            (f["visual_complexity"] for f in feedbacks if f.get("visual_complexity") != "ok"), "ok"
        ),
        "add_text_overlays": any(f.get("add_text_overlays", False) for f in feedbacks),
        "add_motion":        any(f.get("add_motion", False) for f in feedbacks),
        "style_note": " | ".join(
            f["style_note"] for f in feedbacks if f.get("style_note")
        ),
    }


async def _synthesize_feedback(
    judges: list[dict],
    scores: dict,
) -> dict:
    """
    One synthesis call that reads all three panel outputs and produces a single
    coherent, non-contradictory feedback. Uses the brain scores to resolve
    disagreements rather than picking by position or majority count.
    """
    evaluator_block = "\n\n".join(
        f"Evaluator {i+1} ({j['model']}):\n"
        f"  reason: {j['reason']}\n"
        f"  feedback: {json.dumps(j['feedback'])}"
        for i, j in enumerate(judges)
    )

    brain_context = (
        f"Engagement context for this iteration:\n"
        f"  engagement={scores['reward']:.4f}  "
        f"association={scores['association_cortex']:.4f}  "
        f"sensory={scores['primary_sensory']:.4f}\n"
        f"  Peak sensory (passive perception) at t={scores['peak_sensory_second']}s\n"
        f"  Peak engagement (active understanding) at t={scores['peak_memory_second']}s"
    )

    user_msg = f"{brain_context}\n\n{evaluator_block}\n\nSynthesize into one coherent feedback."

    client = _make_client()
    response = await client.chat.completions.create(
        model="claude-3-5-sonnet",
        max_tokens=512,
        messages=[
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
    )
    return json.loads(response.choices[0].message.content.strip())


async def _panel_feedback(
    question: str,
    scores: dict,
    params: dict,
    iteration: int,
    history: list[dict],
    all_frames: list[tuple[int, str]],
) -> dict:
    """
    Three judges run in parallel via TokenRouter, each receiving per-second frames:
      - claude-opus-4-8
      - gpt-4o
      - gemini-1.5-pro

    All three outputs go to a synthesis call that resolves contradictions
    using the brain data as ground truth.
    """
    client      = _make_client()
    text_prompt = _build_text_prompt(question, scores, params, iteration, history)
    messages    = _build_vision_messages(text_prompt, all_frames, scores.get("per_second", []))

    results = await asyncio.gather(
        *[_call_single_judge(m, client, messages) for m in _PANEL_MODELS],
        return_exceptions=True,
    )

    judges = [r for r in results if not isinstance(r, Exception)]
    for model, r in zip(_PANEL_MODELS, results):
        if isinstance(r, Exception):
            logger.warning("[evaluator] judge %s failed: %s", model, r)

    if not judges:
        raise RuntimeError("All panel judges failed — check TOKEN_ROUTER_API_KEY")

    # Synthesize — one model resolves disagreements using brain data as ground truth
    try:
        synthesized = await _synthesize_feedback(judges, scores)
        feedback    = synthesized["feedback"]
        reason      = synthesized["reason"]
    except Exception as e:
        logger.warning("[evaluator] synthesis failed (%s), falling back to naive merge", e)
        feedback = _merge_feedback([j["feedback"] for j in judges])
        reason   = " | ".join(j["reason"] for j in judges)

    return {
        "reason":   reason,
        "feedback": feedback,
        "panel":    judges,
    }


# ── Peer agent coroutine ───────────────────────────────────────────────────────

async def evaluator_agent(question: str, band: Band) -> dict:
    """
    Scores every generated video with TRIBE and sends panel feedback to the
    generator. Runs for all N iterations. Tracks the best-scoring video.

    Returns:
      best_video_path  — path of the highest-reward video seen
      best_reward      — its reward score
      best_scores      — full score dict for the best video
      all_rewards      — reward per iteration (for logging/debugging)
      total_iterations — how many iterations actually ran
    """
    history:        list[dict] = []
    best_video_path: str | None = None
    best_reward:     float = float("-inf")
    best_scores:     dict  = {}
    all_rewards:     list[float] = []

    while True:
        gen_msg = await band.evaluator_recv()

        if gen_msg.get("done"):
            logger.info("[evaluator] all iterations complete")
            break

        video_path = gen_msg["video_path"]
        iteration  = gen_msg["iteration"]
        is_source  = gen_msg.get("is_source", False)
        params     = gen_msg.get("params", {})

        logger.info(
            "[evaluator] %s — scoring %s",
            "source video" if is_source else f"iteration {iteration}",
            video_path,
        )

        # Brain scoring — ground truth
        preds  = await _call_tribe(video_path)
        scores = score_preds(preds)
        reward = scores["reward"]

        # Only track best for generated videos, not the source baseline
        if not is_source:
            all_rewards.append(reward)
            if reward > best_reward:
                best_reward     = reward
                best_video_path = video_path
                best_scores     = scores
                logger.info("[evaluator] new best at iteration %d  reward=%.3f", iteration, reward)
        else:
            logger.info("[evaluator] source video baseline reward=%.3f", reward)

        # Extract frames (one per second, single ffmpeg pass)
        all_frames = await asyncio.to_thread(extract_all_frames_base64, video_path)
        panel      = await _panel_feedback(question, scores, params, iteration, history, all_frames)

        judgment = {
            "reason":   panel["reason"],
            "feedback": panel["feedback"],
            "scores":   scores,
            "panel":    panel["panel"],
        }
        history.append(judgment)

        logger.info(
            "[evaluator] iteration %d  reward=%.3f  (best so far: %.3f)",
            iteration, reward, best_reward,
        )

        await band.evaluator_send(judgment)

    return {
        "best_video_path": best_video_path,
        "best_reward":     best_reward,
        "best_scores":     best_scores,
        "all_rewards":     all_rewards,
        "total_iterations": len(all_rewards),
    }
