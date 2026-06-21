"""
    mp4  ─┐
          ├─► (1) Deepgram transcript + word timestamps
 TRIBE ───┤
  JSON ───┼─► (2) locate the worst "dip zone" from the brain predictions
          │
 user ────┼─► (3) Claude diagnoses *why* attention drops there
 intent   │
          └─► (4) Claude writes a style-locked Pika prompt + a Deepgram
                  TTS narration spec to regenerate that moment.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import anthropic
except Exception:  
    anthropic = None

try:
    from deepgram import DeepgramClient
except Exception: 
    DeepgramClient = None

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception: 
    pass



STYLE_REFERENCE = """
- Flat 2D vector animation, no shading/gradients/3D
- Background: solid white
- Primary colors: light blue (~#5BC8E8) water/accents, yellow (~#F2D94E) fish,
  red (~#E84C3D) callout boxes, navy blue (~#1B2A57) bold text/numbers
- Line style: bold black/dark outlines, rounded shapes, no sharp corners
- Text: bold sans-serif, all-caps in callout boxes, large centered numbers
- Motion: minimal, gentle easing, no camera pans/zooms, no fast cuts
""".strip()

_PLANNER_MODEL = "claude-opus-4-8"


@dataclass
class RetentionPatch:
    """Everything a downstream generator needs to rebuild one weak segment."""

    dip_zone: dict[str, Any]            # {start, end, duration_seconds, reason}
    diagnosis: str                      # why attention drops here
    fix_description: str                # the proposed visual fix, in plain words
    pika_prompt: str                    # ready to send to Pika generate_video
    deepgram_prompt: dict[str, Any]     # {text, model, voice, ...} for Deepgram TTS
    transcript: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Stage 1 — transcription
# ---------------------------------------------------------------------------
def _transcribe(video_path: str) -> list[dict[str, Any]]:
    """Deepgram nova-3 transcription → [{start, end, text}, ...] per utterance.

    Deepgram accepts the mp4 container directly (it demuxes the audio track), so
    no ffmpeg pre-extraction is required.
    """
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not (DeepgramClient and api_key and Path(video_path).is_file()):
        return None

    dg = DeepgramClient(api_key=api_key)
    with open(video_path, "rb") as f:
        audio = f.read()

    resp = dg.listen.v1.media.transcribe_file(
        request=audio,
        model="nova-3",
        smart_format=True,
        utterances=True,
        punctuate=True,
    )

    segments = [
        {"start": round(u.start, 2), "end": round(u.end, 2), "text": u.transcript}
        for u in resp.results.utterances
    ]
    return segments or None


def _format_transcript(segments: list[dict[str, Any]]) -> str:
    lines = []
    for s in segments:
        sm, ss = divmod(s["start"], 60)
        em, es = divmod(s["end"], 60)
        lines.append(f'[{int(sm)}:{ss:05.2f}-{int(em)}:{es:05.2f}] "{s["text"]}"')
    return "\n".join(lines)


def _window_text(segments: list[dict[str, Any]], start: float, end: float) -> str:
    """Concatenate transcript text overlapping [start, end]."""
    hit = [s["text"] for s in segments if s["start"] < end and s["end"] > start]
    return " ".join(hit).strip()


# ---------------------------------------------------------------------------
# Stage 2 — find the dip zone in the TRIBE v2 output
# ---------------------------------------------------------------------------
def _load_tribe(tribe_output: str | dict | list) -> dict | list:
    if isinstance(tribe_output, (dict, list)):
        return tribe_output
    p = Path(tribe_output)
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _find_dip_zone(tribe: dict | list, transcript: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the lowest-engagement window from the TRIBE v2 JSON.

    The TRIBE output is flexible across teammates' formats, so we accept a few
    shapes and degrade gracefully:

      * {"dip_zones": [{"start", "end", ...}]}            -> take the first
      * {"engagement": [floats per second]}               -> argmin window
      * {"scores": [{"second", "reward"|"attention"}]}    -> argmin window
      * anything else                                      -> demo dip zone
    """
    # 1) explicit dip zones already computed upstream
    if isinstance(tribe, dict) and tribe.get("dip_zones"):
        z = tribe["dip_zones"][0]
        start, end = float(z["start"]), float(z["end"])
        return _dip(start, end, transcript, z.get("reason", "flagged by TRIBE v2"))

    # 2) a per-second engagement/attention/reward curve
    curve = _extract_curve(tribe)
    if curve:
        lo = min(range(len(curve)), key=lambda i: curve[i])
        # widen to a ~15s window centred on the trough, clamped to the curve
        start = max(0, lo - 7)
        end = min(len(curve), lo + 8)
        reason = f"lowest predicted engagement (score {curve[lo]:.3f}) at second {lo}"
        return _dip(float(start), float(end), transcript, reason)

    # 3) nothing usable → demo
    return None


def _extract_curve(tribe: dict | list) -> list[float]:
    if isinstance(tribe, dict):
        for key in ("engagement", "attention", "reward", "curve"):
            v = tribe.get(key)
            if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                return [float(x) for x in v]
        scores = tribe.get("scores")
        if isinstance(scores, list) and scores and isinstance(scores[0], dict):
            for k in ("reward", "attention", "engagement", "value"):
                if k in scores[0]:
                    return [float(s.get(k, 0.0)) for s in scores]
    if isinstance(tribe, list) and tribe and isinstance(tribe[0], (int, float)):
        return [float(x) for x in tribe]
    return []


def _dip(start: float, end: float, transcript: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "duration_seconds": round(end - start),
        "transcript": _window_text(transcript, start, end) or None,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Stages 3 & 4 — Claude turns the dip zone + intent into the two prompts
# ---------------------------------------------------------------------------
def _plan_with_llm(
    dip: dict[str, Any],
    transcript: list[dict[str, Any]],
    user_intent: str,
) -> dict[str, Any]:
    """One Claude call returns diagnosis + Pika prompt + Deepgram TTS spec."""
    prompt = f"""You are an educational video doctor. A neuroscience attention model
(TRIBE v2) flagged a low-retention "dip zone" in a lesson. Your job: design a
drop-in replacement clip (visuals + narration) that raises predicted retention.

USER GOAL: {user_intent}

DIP ZONE ({dip['start']:.1f}s–{dip['end']:.1f}s, ~{dip['duration_seconds']}s)
Why it was flagged: {dip['reason']}
What is narrated here: "{dip['transcript']}"

SURROUNDING TRANSCRIPT:
{_format_transcript(transcript)}

LOCKED VISUAL STYLE (the regenerated clip MUST match this — no new colors,
characters, or stylistic elements):
{STYLE_REFERENCE}

Return ONLY valid JSON in exactly this shape:
{{
  "diagnosis": "<1-2 sentences: why attention drops here>",
  "fix_description": "<1-2 sentences: the concrete visual fix>",
  "pika_prompt": "<single continuous prompt for an AI video generator. Open with the locked style as a prefix, then describe ONLY the motion/action of the fix, then end with pacing/duration (~{dip['duration_seconds']}s). Under 200 words, no line breaks.>",
  "deepgram_prompt": {{
    "text": "<the new narration script to speak over the clip — tightened, concrete, retention-optimized; must stay time-aligned to ~{dip['duration_seconds']}s>",
    "model": "aura-2-thalia-en",
    "voice": "warm, upbeat teacher",
    "speed": 1.0
  }}
}}"""

    if anthropic and os.getenv("ANTHROPIC_API_KEY"):
        try:
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=_PLANNER_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text
            return json.loads(text[text.find("{"): text.rfind("}") + 1])
        except Exception:
            pass  # fall through to deterministic stub

    return None


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def generate_retention_patch(
    video_path: str,
    tribe_output: str | dict | list,
    user_intent: str,
) -> dict[str, Any]:
    """Turn a flagged lesson into the prompts that regenerate its weakest moment.

    Args:
        video_path: path to the source lesson .mp4.
        tribe_output: TRIBE v2 prediction output — a path to JSON, or an already
            loaded dict/list (see ``_find_dip_zone`` for accepted shapes).
        user_intent: free text describing what the user wants improved.

    Returns:
        A plain dict (``RetentionPatch.to_dict()``) with ``pika_prompt`` and
        ``deepgram_prompt`` ready to hand to Pika / Deepgram, plus the diagnosis,
        the located dip zone, and the transcript for traceability.
    """
    transcript = _transcribe(video_path)
    tribe = _load_tribe(tribe_output)
    dip = _find_dip_zone(tribe, transcript)
    plan = _plan_with_llm(dip, transcript, user_intent)

    patch = RetentionPatch(
        dip_zone=dip,
        diagnosis=plan["diagnosis"],
        fix_description=plan["fix_description"],
        pika_prompt=plan["pika_prompt"],
        deepgram_prompt=plan["deepgram_prompt"],
        transcript=transcript,
        metadata={
            "video_path": video_path,
            "user_intent": user_intent,
            "planner_model": _PLANNER_MODEL,
            "source_window": f"{dip['start']:.1f}s-{dip['end']:.1f}s",
        },
    )
    return patch.to_dict()

if __name__ == "__main__":
    # Smoke test with no external services: runs entirely on demo fixtures.
    out = generate_retention_patch(
        video_path="uploads/demo_lesson.mp4",
        tribe_output={"engagement": [0.9, 0.8, 0.7, 0.3, 0.2, 0.25, 0.6, 0.8]},
        user_intent="make the division explanation stickier for 4th graders",
    )
    print(json.dumps(out, indent=2))
