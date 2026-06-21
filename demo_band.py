"""
Band AI two-agent demo with fake data.

Generator and Evaluator run as live Band AI agents in your chat room.
They take turns: Generator proposes fake video params, Evaluator responds
with fake brain scores and feedback. 5 iterations, then done.

Setup:
  1. Create two External Agents in Band AI UI (see README below)
  2. Set BAND_GENERATOR_ID, BAND_GENERATOR_KEY, BAND_EVALUATOR_ID,
     BAND_EVALUATOR_KEY, BAND_CHAT_ID in .env
  3. uv run python demo_band.py

To kick off the demo, send this message in your Band AI chat room:
  @Generator please start iteration 1
"""

import asyncio
import logging
import os
import random

from dotenv import load_dotenv

from band import Agent
from band.core.simple_adapter import SimpleAdapter
from band.core.types import PlatformMessage

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

CHAT_ID = os.getenv("BAND_CHAT_ID", "be444dc8-905f-47ac-ad1d-9f774f5159b0")
MAX_ITERATIONS = 5

# ─────────────────────────────────────────────── fake data generators

_STYLES = ["kinetic typography", "2D animation", "diagram walkthrough", "whiteboard sketch"]
_PACING = ["fast", "medium", "slow"]
_COMPLEXITY = ["high", "medium", "low"]


def _fake_params(iteration: int) -> dict:
    return {
        "style": _STYLES[iteration % len(_STYLES)],
        "pacing": _PACING[iteration % len(_PACING)],
        "visual_complexity": _COMPLEXITY[(iteration + 1) % len(_COMPLEXITY)],
        "add_text_overlays": iteration % 2 == 0,
        "add_motion": True,
        "creative_direction": (
            f"Iteration {iteration}: focus on the key moment at second "
            f"{random.randint(2,8)}s where mind-wandering peaked. "
            "Add a visual hook with kinetic text to re-engage attention."
        ),
    }


def _fake_brain_scores(iteration: int) -> dict:
    base = 0.3 + iteration * 0.07  # improves each iteration
    hipp = round(base + random.uniform(-0.05, 0.05), 4)
    pfc  = round(base - 0.1 + random.uniform(-0.05, 0.05), 4)
    amyg = round(base - 0.05 + random.uniform(-0.03, 0.03), 4)
    dmn  = round(0.4 - iteration * 0.05 + random.uniform(-0.03, 0.03), 4)
    reward = round(hipp + pfc + amyg - 2.0 * dmn, 4)
    return {"hippocampus": hipp, "left_pfc": pfc, "amygdala": amyg,
            "dmn": dmn, "reward": reward, "iteration": iteration}


def _fake_feedback(scores: dict) -> str:
    worst_second = random.randint(2, 9)
    if scores["dmn"] > 0.3:
        return (
            f"DMN spike at t={worst_second}s — viewer's mind wandered there. "
            "Add a visual hook or motion cut at that moment. "
            f"Memory regions are low (hipp={scores['hippocampus']:.3f}): "
            "try adding text overlays for key terms."
        )
    return (
        f"Good improvement! Reward={scores['reward']:.4f}. "
        f"Hippocampus at {scores['hippocampus']:.3f} — keep the current pacing "
        "but increase visual complexity to push encoding higher."
    )


# ─────────────────────────────────────────────── Generator adapter

class GeneratorAdapter(SimpleAdapter):
    """
    Responds to any mention by proposing fake video parameters and
    @mentioning Evaluator to score them.
    """

    def __init__(self, evaluator_handle: str):
        super().__init__()
        self._evaluator_handle = evaluator_handle  # e.g. "evaluator" or full handle
        self._iteration = 0

    async def on_message(self, msg: PlatformMessage, tools, history, participants_msg,
                         contacts_msg, *, is_session_bootstrap, room_id):
        content = msg.content.lower()

        # Parse iteration from evaluator feedback or initial trigger
        if "iteration" in content or "start" in content or "score" in content:
            self._iteration += 1

        if self._iteration > MAX_ITERATIONS:
            await tools.send_event(
                content=f"[GENERATOR] All {MAX_ITERATIONS} iterations complete. Best video: iteration_{self._iteration-1}.mp4",
                message_type="task",
                metadata={"total_iterations": MAX_ITERATIONS},
            )
            return

        params = _fake_params(self._iteration)
        fake_video = f"video_iteration_{self._iteration}.mp4"

        await tools.send_event(
            content=f"[GENERATOR] Iteration {self._iteration} — generating video",
            message_type="tool_call",
            metadata={"params": params, "video": fake_video},
        )

        reply = (
            f"**Iteration {self._iteration}/{MAX_ITERATIONS}** — video generated\n\n"
            f"Style: {params['style']} | Pacing: {params['pacing']} | "
            f"Complexity: {params['visual_complexity']}\n"
            f"Text overlays: {params['add_text_overlays']} | Motion: {params['add_motion']}\n\n"
            f"Creative direction: _{params['creative_direction']}_\n\n"
            f"Video: `{fake_video}`\n\n"
            f"@{self._evaluator_handle} please score this video"
        )
        await tools.send_message(reply, mentions=[self._evaluator_handle])
        logger.info("[Generator] Sent iteration %d to evaluator", self._iteration)


# ─────────────────────────────────────────────── Evaluator adapter

class EvaluatorAdapter(SimpleAdapter):
    """
    Responds to any mention with fake TRIBE v2 brain scores and
    @mentions Generator with improvement feedback.
    """

    def __init__(self, generator_handle: str):
        super().__init__()
        self._generator_handle = generator_handle
        self._best_reward = float("-inf")
        self._best_iteration = 0
        self._iteration = 0

    async def on_message(self, msg: PlatformMessage, tools, history, participants_msg,
                         contacts_msg, *, is_session_bootstrap, room_id):
        content = msg.content.lower()

        # Parse which iteration we're scoring
        if "iteration" in content:
            try:
                for word in content.split():
                    if word.isdigit():
                        self._iteration = int(word)
                        break
            except Exception:
                self._iteration += 1
        else:
            self._iteration += 1

        if self._iteration > MAX_ITERATIONS:
            return

        scores = _fake_brain_scores(self._iteration)
        feedback = _fake_feedback(scores)

        if scores["reward"] > self._best_reward:
            self._best_reward = scores["reward"]
            self._best_iteration = self._iteration

        await tools.send_event(
            content=f"[EVALUATOR] TRIBE v2 brain scores — iteration {self._iteration}",
            message_type="tool_result",
            metadata={"scores": scores, "best_so_far": self._best_iteration},
        )

        is_last = self._iteration >= MAX_ITERATIONS
        if is_last:
            reply = (
                f"**FINAL EVALUATION — Iteration {self._iteration}**\n\n"
                f"Brain scores:\n"
                f"  Hippocampus (memory): `{scores['hippocampus']}`\n"
                f"  Left PFC (semantic): `{scores['left_pfc']}`\n"
                f"  Amygdala (emotion): `{scores['amygdala']}`\n"
                f"  DMN (mind-wandering): `{scores['dmn']}`\n"
                f"  **Reward: `{scores['reward']}`**\n\n"
                f"Feedback: {feedback}\n\n"
                f"**Best video: iteration {self._best_iteration} "
                f"(reward={self._best_reward:.4f})**\n\n"
                f"@{self._generator_handle} session complete — "
                f"best-of-{MAX_ITERATIONS} result above"
            )
        else:
            reply = (
                f"**Brain scores — Iteration {self._iteration}/{MAX_ITERATIONS}**\n\n"
                f"  Hippocampus: `{scores['hippocampus']}` | "
                f"PFC: `{scores['left_pfc']}` | "
                f"Amygdala: `{scores['amygdala']}` | "
                f"DMN: `{scores['dmn']}`\n"
                f"  **Reward: `{scores['reward']}`** "
                f"(best so far: iteration {self._best_iteration}, "
                f"{self._best_reward:.4f})\n\n"
                f"Feedback: {feedback}\n\n"
                f"@{self._generator_handle} improve and send iteration {self._iteration + 1}"
            )

        await tools.send_message(reply, mentions=[self._generator_handle])
        logger.info("[Evaluator] Scored iteration %d, reward=%.4f", self._iteration, scores["reward"])


# ─────────────────────────────────────────────── main

async def main():
    gen_id  = os.getenv("BAND_GENERATOR_ID")
    gen_key = os.getenv("BAND_GENERATOR_KEY")
    eva_id  = os.getenv("BAND_EVALUATOR_ID")
    eva_key = os.getenv("BAND_EVALUATOR_KEY")

    if not all([gen_id, gen_key, eva_id, eva_key]):
        print("""
Missing environment variables. Add to .env:

  BAND_GENERATOR_ID=<uuid from Band AI UI>
  BAND_GENERATOR_KEY=band_a_...
  BAND_EVALUATOR_ID=<uuid from Band AI UI>
  BAND_EVALUATOR_KEY=band_a_...
  BAND_CHAT_ID=be444dc8-905f-47ac-ad1d-9f774f5159b0

Then re-run: python demo_band.py
""")
        return

    # Both agents need to know the other's @handle so they can mention each other.
    # Band AI handle = whatever name you gave the agent in the UI, lowercased.
    gen_handle = os.getenv("BAND_GENERATOR_HANDLE", "generator")
    eva_handle = os.getenv("BAND_EVALUATOR_HANDLE", "evaluator")

    generator = Agent.create(
        adapter=GeneratorAdapter(evaluator_handle=eva_handle),
        agent_id=gen_id,
        api_key=gen_key,
    )
    evaluator = Agent.create(
        adapter=EvaluatorAdapter(generator_handle=gen_handle),
        agent_id=eva_id,
        api_key=eva_key,
    )

    logger.info("Both agents running. Go to Band AI and send:")
    logger.info("  @%s please start iteration 1", gen_handle)
    logger.info("Watch the conversation unfold in: https://app.band.ai/chat/%s", CHAT_ID)

    await asyncio.gather(generator.run(), evaluator.run())


if __name__ == "__main__":
    asyncio.run(main())
