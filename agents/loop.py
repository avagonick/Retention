"""
run_loop — orchestrates the generator-discriminator iteration.

Usage:
    from agents import run_loop

    result = await run_loop(
        session_id="abc123",
        question="Why does the mitochondria produce ATP?",
        generate_fn=my_pika_generator,
        max_iterations=5,
    )
    print(result["final_video"], result["iterations"])
"""

import logging
from typing import Callable

from .band import Band
from .generator import GeneratorAgent
from .discriminator import DiscriminatorAgent

logger = logging.getLogger(__name__)


async def run_loop(
    session_id: str,
    question: str,
    generate_fn: Callable,
    max_iterations: int = 5,
) -> dict:
    """
    Runs the generator → discriminator loop until the LLM judge approves or
    max_iterations is reached.

    Returns:
        {
          "final_video": str,       # URL/path to best video
          "iterations": int,        # how many rounds were run
          "approved": bool,         # whether discriminator approved
          "band_path": str,         # path to full band state JSON for inspection
        }
    """
    band = Band(session_id)
    generator = GeneratorAgent(band, generate_fn)
    discriminator = DiscriminatorAgent(band)

    approved = False

    for _ in range(max_iterations):
        band.bump()
        logger.info("[loop] iteration %d / %d", band.iteration, max_iterations)

        # Generate
        video_url = generator.run(question)
        logger.info("[loop] generator → %s", video_url)

        # Evaluate
        judgment = await discriminator.run(question)
        logger.info("[loop] discriminator → %s (%s)", judgment["verdict"], judgment["reason"])

        if judgment["verdict"] == "approve":
            approved = True
            break

    final_video = band.get("current_video")
    band.mark_done(final_video)

    return {
        "final_video": final_video,
        "iterations": band.iteration,
        "approved": approved,
        "band_path": str(band._path),
    }
