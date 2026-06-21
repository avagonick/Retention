"""
Band — async message queues for peer-to-peer agent communication.

Two unidirectional asyncio.Queues:
  generator → evaluator  (band.gen_q)
  evaluator → generator  (band.eval_q)

Each agent awaits its inbox and sends to the other's inbox directly.
No orchestrator drives the sequence — agents block on their queue and
wake up the moment a message arrives.

All messages are also:
  1. Appended to a local JSON log file (band_states/<session_id>.json)
  2. Mirrored to a Band AI chat room as events (fire-and-forget)
     so the conversation is visible live in the browser.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_DIR = Path("band_states")
_DEFAULT_CHAT_ID = "be444dc8-905f-47ac-ad1d-9f774f5159b0"

try:
    from thenvoi_rest import AsyncRestClient
    from thenvoi_rest.types import ChatEventRequest
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False


def _event_type(sender: str, msg: dict) -> str:
    """Map a Band message to a Band AI event type."""
    if msg.get("done"):
        return "task"
    if msg.get("is_source"):
        return "thought"
    if sender == "generator":
        return "tool_call"
    return "tool_result"


def _event_content(sender: str, msg: dict) -> str:
    """Format a Band message as human-readable event content for Band AI."""
    if msg.get("done"):
        return f"[{sender.upper()}] Session complete — all iterations finished."

    if msg.get("is_source"):
        return (
            f"[EVALUATOR] Scoring source video (baseline)\n"
            f"Path: {msg.get('video_path', '?')}"
        )

    iteration = msg.get("iteration", "?")

    if sender == "generator":
        params = msg.get("params", {})
        direction = params.get("creative_direction", "")
        return (
            f"[GENERATOR] Iteration {iteration} — video generated\n"
            f"Style: {params.get('style', '?')}  "
            f"Pacing: {params.get('pacing', '?')}  "
            f"Complexity: {params.get('visual_complexity', '?')}\n"
            f"Direction: {direction}\n"
            f"Path: {msg.get('video_path', '?')}"
        )

    # evaluator
    scores = msg.get("scores", {})
    reward = scores.get("reward", msg.get("reward", "?"))
    reason = msg.get("reason", "")
    if scores:
        return (
            f"[EVALUATOR] Iteration {iteration} — brain scores\n"
            f"Reward: {reward:.4f}  "
            f"Hippocampus: {scores.get('hippocampus', '?'):.4f}  "
            f"PFC: {scores.get('left_pfc', '?'):.4f}  "
            f"Amygdala: {scores.get('amygdala', '?'):.4f}  "
            f"DMN: {scores.get('dmn', '?'):.4f}\n"
            f"Feedback: {reason}"
        )
    return f"[EVALUATOR] Iteration {iteration}\nFeedback: {reason}"


class Band:
    def __init__(self, session_id: str):
        _STATE_DIR.mkdir(exist_ok=True)
        self.session_id = session_id
        self._log_path = _STATE_DIR / f"{session_id}.json"
        self._log: list[dict] = []

        # One queue per direction — unbounded, so sends never block
        self.gen_q: asyncio.Queue = asyncio.Queue()   # generator → evaluator
        self.eval_q: asyncio.Queue = asyncio.Queue()  # evaluator → generator

        # Band AI mirror client (None if SDK missing or no agent key)
        # Requires an agent key (band_a_*) created via the Band AI UI.
        # The user key (band_u_*) only works with the Enterprise Human API.
        agent_key = os.getenv("BAND_AGENT_KEY")
        self._chat_id = os.getenv("BAND_CHAT_ID", _DEFAULT_CHAT_ID)
        self._band_client = (
            AsyncRestClient(api_key=agent_key, base_url="https://app.band.ai")
            if (_SDK_AVAILABLE and agent_key)
            else None
        )
        if self._band_client:
            logger.info("Band AI mirror enabled → chat %s", self._chat_id)
        else:
            logger.info("Band AI mirror disabled (no BAND_API_KEY or SDK missing)")

    # ---------------------------------------------------------------- send/recv

    async def generator_send(self, msg: dict):
        """Generator posts to evaluator's inbox."""
        self._append_log("generator", msg)
        await self.gen_q.put(msg)

    async def evaluator_send(self, msg: dict):
        """Evaluator posts to generator's inbox."""
        self._append_log("evaluator", msg)
        await self.eval_q.put(msg)

    async def generator_recv(self) -> dict:
        """Generator blocks until evaluator sends something."""
        return await self.eval_q.get()

    async def evaluator_recv(self) -> dict:
        """Evaluator blocks until generator sends something."""
        return await self.gen_q.get()

    # --------------------------------------------------------------- logging

    def _append_log(self, sender: str, msg: dict):
        entry = {
            "from": sender,
            "ts": datetime.now(timezone.utc).isoformat(),
            "msg": msg,
        }
        self._log.append(entry)
        self._log_path.write_text(
            json.dumps({"session_id": self.session_id, "messages": self._log}, indent=2)
        )
        # Mirror to Band AI — fire and forget; never blocks agent communication
        if self._band_client:
            asyncio.create_task(self._mirror(sender, msg))

    async def _mirror(self, sender: str, msg: dict):
        """Post one event to the Band AI chat room. Errors are logged, never raised."""
        try:
            # Strip non-scalar values from metadata (video_path kept separately)
            scalar_meta = {
                k: v for k, v in msg.items()
                if not isinstance(v, (dict, list))
            }
            await self._band_client.agent_api_events.create_agent_chat_event(
                self._chat_id,
                event=ChatEventRequest(
                    content=_event_content(sender, msg),
                    message_type=_event_type(sender, msg),
                    metadata={"session_id": self.session_id, "sender": sender, **scalar_meta},
                ),
            )
        except Exception as exc:
            logger.debug("Band AI mirror error: %s", exc)

    @property
    def log_path(self) -> str:
        return str(self._log_path)
