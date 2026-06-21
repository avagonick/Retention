"""
Band AI agents — Generator and Evaluator wired into the web server.

Flow:
  1. Web POST /process  → puts job on _job_queue
  2. Generator picks up job, runs 5 iterations via Band AI chat
  3. Generator @mentions Evaluator each iteration with video path + params
  4. Evaluator scores video (fake here, real TRIBE v2 later), @mentions Generator back
  5. When done, result written to results/{session_id}.json
  6. Web GET /result/{session_id} returns the result

To run standalone (manual trigger via Band AI):
  python band_agents.py

To integrate with web server, import start_agents() and call it at startup.
"""

import asyncio
import json
import logging
import os
import random
import uuid
from pathlib import Path

from dotenv import load_dotenv

from band import Agent
from band.core.simple_adapter import SimpleAdapter
from band.core.types import PlatformMessage

load_dotenv()

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MAX_ITERATIONS = 5
BAND_CHAT_ID = os.getenv("BAND_CHAT_ID", "be444dc8-905f-47ac-ad1d-9f774f5159b0")

# Shared job queue — web routes put jobs here, Generator picks them up
_job_queue: asyncio.Queue = asyncio.Queue()


def submit_job(session_id: str, question: str, source_video_path: str) -> None:
    """Called by /process route to kick off the agent loop."""
    _job_queue.put_nowait({
        "session_id": session_id,
        "question": question,
        "source_video_path": source_video_path,
    })
    logger.info("[band_agents] job submitted: session=%s question=%r", session_id, question)


def get_result(session_id: str) -> dict | None:
    """Called by /result route — returns result or None if still running."""
    path = RESULTS_DIR / f"{session_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


# ─────────────────────────────────────────── fake data (swap for real later)

_STYLES = ["kinetic typography", "2D animation", "diagram walkthrough", "whiteboard sketch"]
_PACING = ["fast", "medium", "slow"]


def _fake_params(iteration: int) -> dict:
    return {
        "style": _STYLES[iteration % len(_STYLES)],
        "pacing": _PACING[iteration % len(_PACING)],
        "visual_complexity": ["high", "medium", "low"][(iteration + 1) % 3],
        "creative_direction": (
            f"Iteration {iteration}: focus on second {random.randint(2, 8)}s "
            "where DMN peaked. Add kinetic text to re-engage attention."
        ),
    }


def _fake_brain_scores(iteration: int) -> dict:
    base = 0.3 + iteration * 0.07
    hipp = round(base + random.uniform(-0.05, 0.05), 4)
    pfc  = round(base - 0.1 + random.uniform(-0.05, 0.05), 4)
    amyg = round(base - 0.05 + random.uniform(-0.03, 0.03), 4)
    dmn  = round(0.4 - iteration * 0.05 + random.uniform(-0.03, 0.03), 4)
    return {
        "hippocampus": hipp, "left_pfc": pfc,
        "amygdala": amyg, "dmn": dmn,
        "reward": round(hipp + pfc + amyg - 2.0 * dmn, 4),
    }


# ─────────────────────────────────────────────────────── Generator

class GeneratorAdapter(SimpleAdapter):
    """
    Listens for two things:
      1. Web job (via _job_queue) → starts iteration 1 → @mentions Evaluator
      2. Evaluator feedback → generates next iteration → @mentions Evaluator again

    Stores `tools` on every on_message so the job runner can send messages
    outside the on_message call.
    """

    def __init__(self, evaluator_handle: str):
        super().__init__()
        self._evaluator_handle = evaluator_handle
        self._tools = None
        self._room_id = None
        self._job: dict | None = None
        self._iteration = 0
        self._all_rewards: list[float] = []

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        await super().on_started(agent_name, agent_description)
        asyncio.create_task(self._job_runner(), name="generator-job-runner")
        logger.info("[Generator] started, waiting for web jobs or @mentions")

    async def _job_runner(self) -> None:
        """Background task: waits for web-submitted jobs and kicks off iteration 1."""
        while True:
            job = await _job_queue.get()
            # Wait until we have a tools reference (agent is in a room)
            while self._tools is None:
                await asyncio.sleep(0.5)
            self._job = job
            self._iteration = 0
            self._all_rewards = []
            await self._next_iteration(reason="Source video — starting from scratch")

    async def _next_iteration(self, reason: str = "") -> None:
        self._iteration += 1
        if self._iteration > MAX_ITERATIONS:
            return  # evaluator will handle final done signal

        params = _fake_params(self._iteration)
        # In the real version: video_path = generate(question, prev_video, iteration, reason, **params)
        fake_video = f"video_iter_{self._iteration}_{self._job['session_id'][:8]}.mp4"

        await self._tools.send_event(
            content=f"[GENERATOR] Generating iteration {self._iteration}/{MAX_ITERATIONS}",
            message_type="tool_call",
            metadata={"params": params, "video": fake_video, "session": self._job["session_id"]},
        )

        msg = (
            f"**Iteration {self._iteration}/{MAX_ITERATIONS}**\n"
            f"Question: _{self._job['question']}_\n\n"
            f"Style: {params['style']} | Pacing: {params['pacing']} | "
            f"Complexity: {params['visual_complexity']}\n"
            f"Direction: _{params['creative_direction']}_\n\n"
            f"Video: `{fake_video}`\n\n"
            f"@{self._evaluator_handle} please score this"
        )
        await self._tools.send_message(msg, mentions=[self._evaluator_handle])

    async def on_message(self, msg: PlatformMessage, tools, history, participants_msg,
                         contacts_msg, *, is_session_bootstrap, room_id) -> None:
        self._tools = tools
        self._room_id = room_id

        content = msg.content.lower()

        # Manual trigger from Band AI chat (no web job in flight)
        if self._job is None and ("start" in content or "iteration 1" in content):
            self._job = {
                "session_id": str(uuid.uuid4()),
                "question": "manually triggered from Band AI",
                "source_video_path": "none",
            }
            self._iteration = 0
            self._all_rewards = []
            await self._next_iteration()
            return

        # Evaluator feedback → next iteration
        if self._job and ("score" in content or "reward" in content or "feedback" in content):
            # Parse reward from message if present
            for word in msg.content.replace(":", " ").split():
                try:
                    val = float(word.strip("`*"))
                    if -5 < val < 5:
                        self._all_rewards.append(val)
                        break
                except ValueError:
                    pass

            if self._iteration >= MAX_ITERATIONS:
                # Final — write result
                best_reward = max(self._all_rewards) if self._all_rewards else 0.0
                best_iter = self._all_rewards.index(best_reward) + 1 if self._all_rewards else 1
                result = {
                    "session_id": self._job["session_id"],
                    "question": self._job["question"],
                    "best_iteration": best_iter,
                    "best_reward": best_reward,
                    "all_rewards": self._all_rewards,
                    "final_video": f"video_iter_{best_iter}_{self._job['session_id'][:8]}.mp4",
                    "status": "complete",
                }
                (RESULTS_DIR / f"{self._job['session_id']}.json").write_text(
                    json.dumps(result, indent=2)
                )
                await tools.send_message(
                    f"**Session complete** — best video: iteration {best_iter} "
                    f"(reward={best_reward:.4f})\n\n"
                    f"Result saved → results/{self._job['session_id']}.json",
                    mentions=[self._evaluator_handle],
                )
                logger.info("[Generator] session %s complete", self._job["session_id"])
                self._job = None
            else:
                await self._next_iteration(reason=msg.content)


# ─────────────────────────────────────────────────────── Evaluator

class EvaluatorAdapter(SimpleAdapter):
    """
    Scores each video the Generator sends, responds with brain scores + feedback.
    In the real version: calls TRIBE v2 on Lightning AI + runs LLM panel.
    """

    def __init__(self, generator_handle: str):
        super().__init__()
        self._generator_handle = generator_handle
        self._best_reward = float("-inf")
        self._best_iteration = 0

    async def on_message(self, msg: PlatformMessage, tools, history, participants_msg,
                         contacts_msg, *, is_session_bootstrap, room_id) -> None:
        content = msg.content

        # Parse iteration number
        iteration = 1
        for word in content.split():
            if word.startswith("**Iteration"):
                try:
                    iteration = int(content.split("**Iteration")[1].split("/")[0].strip())
                except Exception:
                    pass
                break

        # Score the video
        scores = _fake_brain_scores(iteration)  # swap for _call_tribe(video_path)

        if scores["reward"] > self._best_reward:
            self._best_reward = scores["reward"]
            self._best_iteration = iteration

        await tools.send_event(
            content=f"[EVALUATOR] TRIBE v2 brain scores — iteration {iteration}",
            message_type="tool_result",
            metadata={"scores": scores, "best_so_far": self._best_iteration},
        )

        is_last = iteration >= MAX_ITERATIONS
        footer = (
            f"**Best so far: iteration {self._best_iteration} "
            f"(reward={self._best_reward:.4f})**"
        )
        dmn_note = (
            f"DMN spike at t={random.randint(2, 9)}s — add visual hook there. "
            if scores["dmn"] > 0.3
            else "Good attention retention — push memory encoding higher. "
        )

        reply = (
            f"**Brain scores — Iteration {iteration}/{MAX_ITERATIONS}**\n"
            f"Hippocampus: `{scores['hippocampus']}` | "
            f"PFC: `{scores['left_pfc']}` | "
            f"Amygdala: `{scores['amygdala']}` | "
            f"DMN: `{scores['dmn']}`\n"
            f"**Reward: `{scores['reward']}`** {footer}\n\n"
            f"Feedback: {dmn_note}\n\n"
            f"@{self._generator_handle} "
            + ("session complete — you have all scores" if is_last
               else f"improve and send iteration {iteration + 1}")
        )
        await tools.send_message(reply, mentions=[self._generator_handle])
        logger.info("[Evaluator] scored iteration %d, reward=%.4f", iteration, scores["reward"])


# ─────────────────────────────────────────────────────── startup

async def start_agents() -> None:
    """Start both agents. Call this at web server startup."""
    gen_id  = os.getenv("BAND_GENERATOR_ID")
    gen_key = os.getenv("BAND_GENERATOR_KEY")
    eva_id  = os.getenv("BAND_EVALUATOR_ID")
    eva_key = os.getenv("BAND_EVALUATOR_KEY")
    gen_handle = os.getenv("BAND_GENERATOR_HANDLE", "generator")
    eva_handle = os.getenv("BAND_EVALUATOR_HANDLE", "evaluator")

    if not all([gen_id, gen_key, eva_id, eva_key]):
        logger.warning("[band_agents] Missing BAND_GENERATOR_ID/KEY or BAND_EVALUATOR_ID/KEY — agents not started")
        return

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

    logger.info("[band_agents] Generator + Evaluator connected to Band AI")
    await asyncio.gather(generator.run(), evaluator.run())


# ─────────────────────────────────────────────────────── standalone entry point

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    asyncio.run(start_agents())
