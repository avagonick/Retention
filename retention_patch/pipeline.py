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
    from deepgram import DeepgramClient, SpeakOptions
except Exception:
    DeepgramClient = None
    SpeakOptions = None

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
import time
import uuid

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

try:
    import requests
except ImportError:
    requests = None

PIKA_API_BASE = os.getenv("PIKA_API_BASE", "https://api.pika.art/v1")
_PIKA_MODEL = "pika-2.2"
_PIKA_POLL_INTERVAL = 5
_PIKA_POLL_TIMEOUT = 600

def _call_pika_api(prompt: str) -> str | None:
    """Calls the Pika API to generate a video and returns the path to the video file.

    Submits a text-to-video job, polls until it finishes, then downloads the
    rendered mp4 to a temp path.
    """
    api_key = os.getenv("PIKA_API_KEY")
    if not (requests and api_key):
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        # 1) submit the generation job
        resp = requests.post(
            f"{PIKA_API_BASE}/generate",
            headers=headers,
            json={
                "model": _PIKA_MODEL,
                "promptText": prompt,
                "aspectRatio": "16:9",
                "resolution": "1080p",
            },
            timeout=30,
        )
        resp.raise_for_status()
        job_id = resp.json()["id"]

        # 2) poll until the render completes
        deadline = time.time() + _PIKA_POLL_TIMEOUT
        video_url = None
        while time.time() < deadline:
            status_resp = requests.get(
                f"{PIKA_API_BASE}/generate/{job_id}",
                headers=headers,
                timeout=30,
            )
            status_resp.raise_for_status()
            data = status_resp.json()
            status = data.get("status")
            if status == "finished":
                video_url = data.get("videoUrl") or data.get("url")
                break
            if status in ("failed", "cancelled"):
                return None
            time.sleep(_PIKA_POLL_INTERVAL)

        if not video_url:
            return None

        # 3) download the rendered clip
        video_path = f"/tmp/pika_video_{uuid.uuid4()}.mp4"
        with requests.get(video_url, stream=True, timeout=120) as dl:
            dl.raise_for_status()
            with open(video_path, "wb") as f:
                for chunk in dl.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
        return video_path
    except Exception:
        return None

def _call_deepgram_tts_api(prompt: dict[str, Any]) -> str | None:
    """Calls the Deepgram Aura TTS API to synthesize narration audio.

    Returns the path to the rendered mp3 file.
    """
    api_key = os.getenv("DEEPGRAM_API_KEY")
    text = (prompt or {}).get("text", "").strip()
    if not (DeepgramClient and SpeakOptions and api_key and text):
        return None

    try:
        dg = DeepgramClient(api_key)
        options = SpeakOptions(
            model=prompt.get("model", "aura-2-thalia-en"),
            encoding="mp3",
            sample_rate=24000,
        )
        audio_path = f"/tmp/deepgram_audio_{uuid.uuid4()}.mp3"
        response = dg.speak.rest.v("1").save(
            audio_path,
            {"text": text},
            options,
        )
        return getattr(response, "filename", audio_path)
    except Exception:
        return None

def _combine_video_and_audio(video_path: str, audio_path: str) -> str | None:
    """Mux the narration audio onto the video track; return the combined path.

    Uses a stream mux (video from input 0, audio from input 1), NOT concat —
    concat would play them one-after-another instead of overlaying the audio.
    """
    if not (ffmpeg and video_path and audio_path):
        return None

    output_path = f"/tmp/combined_video_{uuid.uuid4()}.mp4"
    try:
        in_v = ffmpeg.input(video_path)
        in_a = ffmpeg.input(audio_path)
        (
            ffmpeg
            .output(in_v.video, in_a.audio, output_path,
                    vcodec="copy", acodec="aac", shortest=None)
            .run(overwrite_output=True, quiet=True)
        )
        return output_path
    except Exception:
        return None

def generate_retention_patch(
    video_path: str,
    tribe_output: str | dict | list,
    user_intent: str,
) -> str | None:
    """
    Turn a flagged lesson into a regenerated, improved MP4.

    Args:
        video_path: path to the source lesson .mp4.
        tribe_output: TRIBE v2 prediction output — a path to JSON, or an already
            loaded dict/list (see ``_find_dip_zone`` for accepted shapes).
        user_intent: free text describing what the user wants improved.

    Returns:
        A path to the newly generated mp4 file, or None if an error occurred.
    """
    transcript = _transcribe(video_path)
    if not transcript:
        return None

    tribe = _load_tribe(tribe_output)
    dip = _find_dip_zone(tribe, transcript)
    if not dip:
        return None

    plan = _plan_with_llm(dip, transcript, user_intent)
    if not plan:
        return None

    # Generate video and audio
    generated_video_path = _call_pika_api(plan["pika_prompt"])
    generated_audio_path = _call_deepgram_tts_api(plan["deepgram_prompt"])

    if not generated_video_path or not generated_audio_path:
        return None

    # Combine video and audio
    combined_video_path = _combine_video_and_audio(generated_video_path, generated_audio_path)

    # Clean up intermediate files
    if generated_video_path and os.path.exists(generated_video_path):
        os.remove(generated_video_path)
    if generated_audio_path and os.path.exists(generated_audio_path):
        os.remove(generated_audio_path)

    return combined_video_path


if __name__ == "__main__":
    # Smoke test with no external services: runs entirely on demo fixtures.
    out_path = generate_retention_patch(
        video_path="uploads/demo_lesson.mp4",
        tribe_output={"engagement": [0.9, 0.8, 0.7, 0.3, 0.2, 0.25, 0.6, 0.8]},
        user_intent="make the division explanation stickier for 4th graders",
    )
    if out_path:
        print(f"Generated video: {out_path}")
    else:
        print("Failed to generate video.")
