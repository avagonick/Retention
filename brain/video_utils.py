"""Small video helpers for the local pipeline (uses the ffmpeg bundled with
imageio-ffmpeg, so no separate ffmpeg install is required)."""

from __future__ import annotations

import subprocess

import imageio_ffmpeg


def ffmpeg_exe() -> str:
    """Path to the ffmpeg binary bundled with imageio-ffmpeg."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def pad_video_tail(in_path: str, out_path: str, seconds: float = 6.0) -> str:
    """Append `seconds` of frozen last frame + silence to the end of a clip.

    Why: TRIBE returns ~1 BOLD row per second of video and stops at the clip
    end, so the hemodynamic response to the final ~5 s of real content never
    appears in a returned row. Scoring a tail-padded copy gives those rows real
    values; the padding's own rows (stimulus >= original end) are then discarded.

    The freeze-frame tail is neutral (no new visual/audio events), so it barely
    perturbs the response to the real ending while extending coverage.
    """
    vf = f"tpad=stop_mode=clone:stop_duration={seconds}"
    af = f"apad=pad_dur={seconds}"
    cmd = [
        ffmpeg_exe(), "-y", "-i", in_path,
        "-vf", vf, "-af", af,
        "-c:v", "libx264", "-c:a", "aac",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Retry video-only (clip may have no audio stream).
        cmd_noaudio = [
            ffmpeg_exe(), "-y", "-i", in_path, "-vf", vf,
            "-c:v", "libx264", "-an", out_path,
        ]
        proc2 = subprocess.run(cmd_noaudio, capture_output=True, text=True)
        if proc2.returncode != 0:
            raise RuntimeError(
                "ffmpeg tail-pad failed:\n" + proc.stderr[-800:] + "\n" + proc2.stderr[-800:]
            )
    return out_path
