"""
Band AI agents — Generator and Evaluator wired into the web server.

Flow:
  1. Web POST /process  → puts job on _job_queue
  2. GeneratorAdapter._job_runner picks up the job
  3. Posts opening message to Band AI, then calls run_loop() from agents/loop.py
  4. run_loop() runs the real generator_agent + evaluator_agent concurrently:
       - generator_agent  → Claude reasons over brain scores, calls generate_fn
       - evaluator_agent  → TRIBE v2 brain scoring + 3-judge LLM panel feedback
       - agents/band.py   → mirrors each queue message as a Band AI event
  5. When run_loop() finishes, result written to results/{session_id}.json
  6. Web GET /result/{session_id} returns the result

generate_fn is currently a placeholder (returns a fake path).
Swap it for the real Pika call when ready.
"""

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from band import Agent
from band.core.simple_adapter import SimpleAdapter
from band.core.types import PlatformMessage

from agents.loop import run_loop
from generate import generate as pika_generate_fn

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




# ─────────────────────────────────────────────────────── Generator Band AI agent

class GeneratorAdapter(SimpleAdapter):
    """
    Band AI-visible agent that initiates sessions and posts results.
    The real work runs inside run_loop() → generator_agent() + evaluator_agent().
    """

    def __init__(self, evaluator_handle: str, evaluator_id: str, chat_id: str, agent_key: str):
        super().__init__()
        self._evaluator_handle = evaluator_handle
        self._evaluator_id = evaluator_id
        self._chat_id = chat_id
        self._agent_key = agent_key

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        await super().on_started(agent_name, agent_description)
        asyncio.create_task(self._job_runner(), name="gen-job-runner")
        logger.info("[Generator] started — waiting for jobs")

    async def _job_runner(self) -> None:
        from thenvoi_rest import AsyncRestClient
        from thenvoi_rest.types import ChatMessageRequest, ChatMessageRequestMentionsItem

        client = AsyncRestClient(api_key=self._agent_key, base_url="https://app.band.ai")

        await asyncio.sleep(5)  # wait for WebSocket subscriptions

        while True:
            job = await _job_queue.get()
            session_id = job["session_id"]
            question   = job["question"]
            source     = job["source_video_path"]

            # Post opening message to Band AI
            try:
                await client.agent_api_messages.create_agent_chat_message(
                    self._chat_id,
                    message=ChatMessageRequest(
                        content=(
                            f"**New session — {session_id[:8]}**\n"
                            f"Question: _{question}_\n\n"
                            f"Source video: `{source}`\n\n"
                            f"Running {MAX_ITERATIONS} iterations — "
                            f"@{self._evaluator_handle} will score each one with TRIBE v2."
                        ),
                        mentions=[ChatMessageRequestMentionsItem(
                            id=self._evaluator_id,
                            handle=self._evaluator_handle,
                        )],
                    ),
                )
            except Exception as exc:
                logger.error("[Generator] failed to post opening message: %s", exc)

            # Run the real generator + evaluator loop
            try:
                result = await run_loop(
                    session_id=session_id,
                    question=question,
                    source_video_path=source,
                    generate_fn=pika_generate_fn,
                    max_iterations=MAX_ITERATIONS,
                )
            except Exception as exc:
                logger.error("[Generator] run_loop failed: %s", exc)
                continue

            # Save result for /result polling
            (RESULTS_DIR / f"{session_id}.json").write_text(
                json.dumps({**result, "session_id": session_id, "status": "complete"}, indent=2)
            )

            # Post completion summary to Band AI
            try:
                await client.agent_api_messages.create_agent_chat_message(
                    self._chat_id,
                    message=ChatMessageRequest(
                        content=(
                            f"**Session complete — {session_id[:8]}**\n\n"
                            f"Best iteration: {result['best_iteration']} of {result['total_iterations']}\n"
                            f"Best reward: `{result['best_reward']:.4f}`\n"
                            f"All rewards: {[round(r, 4) for r in result['all_rewards']]}\n\n"
                            f"Best video: `{result['final_video']}`"
                        ),
                    ),
                )
            except Exception as exc:
                logger.error("[Generator] failed to post completion: %s", exc)

            logger.info("[Generator] session %s complete — best iter %d reward=%.4f",
                        session_id, result["best_iteration"], result["best_reward"])

    async def on_message(self, msg: PlatformMessage, tools, history, participants_msg,
                         contacts_msg, *, is_session_bootstrap, room_id) -> None:
        # Sessions are initiated by _job_runner from the web input.
        # This handles manual triggers from the Band AI chat UI only.
        content = msg.content.lower()
        if "start" in content or "run" in content:
            submit_job(
                session_id="manual-" + str(uuid.uuid4())[:8],
                question=msg.content,
                source_video_path="uploads/source.mp4",
            )


# ─────────────────────────────────────────────────────── Evaluator Band AI agent

class EvaluatorAdapter(SimpleAdapter):
    """
    Band AI presence for the Evaluator.
    Real evaluation runs inside run_loop() → evaluator_agent() → TRIBE v2 + LLM panel.
    This adapter exists so the Evaluator appears as a participant in the chat room.
    """

    def __init__(self, generator_handle: str):
        super().__init__()
        self._generator_handle = generator_handle

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        await super().on_started(agent_name, agent_description)
        logger.info("[Evaluator] connected — real scoring runs via TRIBE v2 inside run_loop()")

    async def on_message(self, msg: PlatformMessage, tools, history, participants_msg,
                         contacts_msg, *, is_session_bootstrap, room_id) -> None:
        pass  # driven by run_loop() internally, not by Band AI messages


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
        adapter=GeneratorAdapter(
            evaluator_handle=eva_handle,
            evaluator_id=eva_id,
            chat_id=BAND_CHAT_ID,
            agent_key=gen_key,
        ),
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
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    if "--test" in sys.argv:
        submit_job(
            session_id="test-" + str(uuid.uuid4())[:8],
            question="How does the hippocampus encode long-term memories?",
            source_video_path="uploads/source.mp4",
        )

    asyncio.run(start_agents())
