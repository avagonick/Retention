"""
DiscriminatorAgent — evaluates generated videos using brain data + LLM judgment.

Flow per iteration:
  1. Read current video local path from band
  2. POST video file to TRIBE on Lightning AI → get fMRI predictions (T × 20484)
  3. Score predictions against memory / DMN / attention vertex masks
  4. Ask Claude to synthesize scores + question → verdict + feedback
  5. Post result to band

Set env vars:
  TRIBE_ENDPOINT   — Lightning AI inference URL (e.g. https://<studio>.lightning.ai/predict)
  ANTHROPIC_API_KEY
"""

import json
import os
from pathlib import Path

import anthropic
import httpx
import numpy as np

from brain.atlas import score_preds
from .band import Band

TRIBE_ENDPOINT = os.getenv("TRIBE_ENDPOINT", "")
APPROVE_THRESHOLD = float(os.getenv("APPROVE_THRESHOLD", "0.3"))
_JUDGE_MODEL = "claude-opus-4-8"


async def _call_tribe(video_path: str) -> np.ndarray:
    """Read video from local path and POST as multipart to Lightning AI TRIBE endpoint."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    async with httpx.AsyncClient(timeout=300) as client:
        with path.open("rb") as f:
            resp = await client.post(
                TRIBE_ENDPOINT,
                files={"video": (path.name, f, "video/mp4")},
            )
        resp.raise_for_status()
        body = resp.json()

    # Expected response: {"shape": [T, 20484], "data": [[...], ...]}
    data = np.array(body["data"], dtype=np.float32)
    return data


def _build_judge_prompt(question: str, scores: dict, iteration: int, history: list[dict]) -> str:
    prior = ""
    if iteration > 1:
        past = [
            f"  iter {m['iteration']}: {m['content'].get('judgment', {}).get('reason', '')}"
            for m in history
            if m["role"] == "discriminator" and m.get("content", {}).get("judgment")
        ]
        if past:
            prior = "\nPrior iteration feedback:\n" + "\n".join(past)

    return f"""You are evaluating an AI-generated educational video for cognitive retention quality.

Question being taught: "{question}"
Iteration: {iteration}

fMRI brain activation scores (fsaverage5 surface, mean over time):
  Memory regions:   {scores['memory']:.4f}   (higher = better encoding)
  DMN (wandering):  {scores['dmn']:.4f}   (lower = better focus)
  Attention network:{scores['attention']:.4f}   (higher = better engagement)
  Composite reward: {scores['reward']:.4f}   (target > {APPROVE_THRESHOLD})
{prior}

Based on the brain scores, reason about what the video is doing right or wrong, then
return ONLY valid JSON in this exact shape (no markdown, no extra keys):

{{
  "verdict": "approve" | "iterate",
  "reason": "<one sentence>",
  "feedback": {{
    "pacing": "faster" | "slower" | "ok",
    "visual_complexity": "increase" | "decrease" | "ok",
    "add_text_overlays": true | false,
    "add_motion": true | false,
    "slow_at_second": <int or null>,
    "style_note": "<brief instruction for the generator, or empty string>"
  }}
}}

Use "approve" only when reward > {APPROVE_THRESHOLD}."""


async def _llm_judge(question: str, scores: dict, iteration: int, history: list[dict]) -> dict:
    client = anthropic.AsyncAnthropic()
    prompt = _build_judge_prompt(question, scores, iteration, history)
    msg = await client.messages.create(
        model=_JUDGE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    return json.loads(text)


class DiscriminatorAgent:
    def __init__(self, band: Band):
        self.band = band

    async def run(self, question: str) -> dict:
        video_path = self.band.get("current_video")
        if not video_path:
            raise ValueError("No current_video on band — run generator first")

        preds = await _call_tribe(video_path)
        scores = score_preds(preds)

        judgment = await _llm_judge(
            question=question,
            scores=scores,
            iteration=self.band.iteration,
            history=self.band.history(),
        )

        self.band.post("discriminator", {
            "scores": scores,
            "judgment": judgment,
            "video_path": video_path,
        })

        return judgment
