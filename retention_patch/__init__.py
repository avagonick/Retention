"""
retention_patch --- turns a flagged video into the prompts
needed to regenerate its weakest moment for better cognitive retention.

    from retention_patch import generate_retention_patch

    patch = generate_retention_patch(
        video_path="uploads/lesson.mp4",
        tribe_output="out/preds.json",      # path or dict (TRIBE v2 JSON)
        user_intent="make the division explanation stickier for 4th graders",
    )

    patch["pika_prompt"]      # -> str, feed to Pika generate_video
    patch["deepgram_prompt"]  # -> {"text", "model", ...}, feed to Deepgram TTS
"""

from .pipeline import RetentionPatch, generate_retention_patch

__all__ = ["RetentionPatch", "generate_retention_patch"]
