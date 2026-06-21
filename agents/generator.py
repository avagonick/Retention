"""
GeneratorAgent — wraps your existing generate_fn as a band-aware agent.

Expected generate_fn signature:
    def generate_fn(question: str, feedback: dict | None, iteration: int) -> str
        → returns a local file path to the saved video (e.g. "uploads/abc.mp4")

The agent reads the latest discriminator feedback from the band before calling
generate_fn, then posts the result back to the band.
"""

from typing import Callable

from .band import Band


class GeneratorAgent:
    def __init__(self, band: Band, generate_fn: Callable):
        self.band = band
        self.generate_fn = generate_fn

    def run(self, question: str) -> str:
        feedback = self.band.latest_from("discriminator")

        video_url = self.generate_fn(
            question=question,
            feedback=feedback,
            iteration=self.band.iteration,
        )

        self.band.post("generator", {
            "video_url": video_url,
            "question": question,
        })
        self.band.set("current_video", video_url)

        return video_url
