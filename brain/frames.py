"""
Extract video frames for vision-capable LLM judges.

Uses ffmpeg. Returns base64-encoded JPEGs for OpenAI-compatible vision APIs.
"""

import base64
import subprocess
import tempfile
from pathlib import Path


def extract_all_frames_base64(video_path: str) -> list[tuple[int, str]]:
    """
    Extract one frame per second from the video in a single ffmpeg pass.

    Frames are scaled to 640px wide (sufficient for layout analysis, keeps
    each frame ~20-40KB). Returns a list of (second, base64_jpeg) pairs
    in chronological order.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", "fps=1,scale=640:-1",
                "-q:v", "5",
                f"{tmpdir}/frame_%04d.jpg",
            ],
            check=True,
            capture_output=True,
        )
        frames = []
        for i, jpg in enumerate(sorted(Path(tmpdir).glob("frame_*.jpg"))):
            frames.append((i, base64.b64encode(jpg.read_bytes()).decode()))
        return frames
