"""
Adapter between the generator agent and retention_patch/pipeline.py.

The generator agent (Claude) decides creative params (style, pacing, etc.)
and writes a concrete creative_direction. This function takes those params,
builds a Pika prompt and a Deepgram TTS spec, then calls the pipeline's
Pika → TTS → combine steps.

The TRIBE dip-finding and Claude planning steps inside generate_retention_patch
are intentionally skipped here — the generator agent's Claude already handles
that reasoning and outputs it as creative_direction.
"""

from retention_patch.pipeline import (
    STYLE_REFERENCE,
    _call_deepgram_tts_api,
    _call_pika_api,
    _combine_video_and_audio,
)


def generate(
    question: str,
    previous_video_path: str,
    iteration: int,
    feedback: str,
    style: str,
    pacing: str,
    visual_complexity: str,
    add_text_overlays: bool,
    add_motion: bool,
    creative_direction: str,
) -> str | None:
    """
    Generate one educational video using Pika + Deepgram TTS.

    Args:
        question:             Learner's question (from Deepgram STT).
        previous_video_path:  Last generated video (source video on iteration 1).
        iteration:            Current iteration number (1-5).
        feedback:             Evaluator's improvement instruction (empty on iteration 1).
        style:                Visual style chosen by the generator agent.
        pacing:               "fast" | "medium" | "slow"
        visual_complexity:    "high" | "medium" | "low"
        add_text_overlays:    Whether to add key-term text overlays.
        add_motion:           Whether to add kinetic animated elements.
        creative_direction:   Concrete brief from the generator agent Claude.

    Returns:
        Local path to the generated .mp4, or None if generation failed.
    """
    overlays_desc = "with text overlays highlighting key terms" if add_text_overlays else "clean visuals, no text overlays"
    motion_desc   = "with kinetic animated elements" if add_motion else "with minimal, gentle motion"
    feedback_line = f"Improvement from last iteration: {feedback}" if feedback else ""

    pika_prompt = (
        f"{STYLE_REFERENCE}\n\n"
        f"Style: {style}. "
        f"Pacing: {pacing}. "
        f"Visual complexity: {visual_complexity}. "
        f"{overlays_desc}. "
        f"{motion_desc}.\n\n"
        f"Topic: {question}\n"
        f"{feedback_line}\n"
        f"Creative direction: {creative_direction}"
    ).strip()

    tts_spec = {
        "text": creative_direction,
        "model": "aura-2-thalia-en",
        "voice": "warm, upbeat teacher",
        "speed": 1.0,
    }

    video_path = _call_pika_api(pika_prompt)
    if not video_path:
        return None

    audio_path = _call_deepgram_tts_api(tts_spec)
    if not audio_path:
        return video_path

    combined = _combine_video_and_audio(video_path, audio_path)
    return combined or video_path
