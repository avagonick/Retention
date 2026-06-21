"""
run_loop — launches generator and discriminator as concurrent peer agents.

Best-of-5 strategy: always run all max_iterations, score each with TRIBE,
return the video that produced the highest brain reward.

No threshold, no stopping condition — the discriminator scores every
iteration and tracks the best. loop.py just collects the result.
"""

import asyncio
import logging
from typing import Callable

from .band import Band
from .generator import generator_agent
from .discriminator import discriminator_agent

logger = logging.getLogger(__name__)


async def run_loop(
    session_id: str,
    question: str,
    source_video_path: str,
    generate_fn: Callable,
    max_iterations: int = 5,
) -> dict:
    band = Band(session_id)

    logger.info("[loop] starting session %s — %d iterations — '%s'", session_id, max_iterations, question)

    gen_task = asyncio.create_task(
        generator_agent(question, source_video_path, band, generate_fn, max_iterations),
        name="generator",
    )
    disc_task = asyncio.create_task(
        discriminator_agent(question, band),
        name="discriminator",
    )

    _, disc_result = await asyncio.gather(gen_task, disc_task)

    best_video  = disc_result["best_video_path"]
    best_reward = disc_result["best_reward"]
    all_rewards = disc_result["all_rewards"]
    best_iter   = all_rewards.index(best_reward) + 1

    logger.info(
        "[loop] done — best iteration %d/%d  reward=%.3f  path=%s",
        best_iter, max_iterations, best_reward, best_video,
    )

    return {
        "final_video":      best_video,
        "best_reward":      best_reward,
        "best_iteration":   best_iter,
        "all_rewards":      all_rewards,
        "total_iterations": disc_result["total_iterations"],
        "band_log":         band.log_path,
    }
