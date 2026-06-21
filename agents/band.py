"""
Band — async message queues for peer-to-peer agent communication.

Two unidirectional asyncio.Queues:
  generator  → discriminator  (band.gen_q)
  discriminator → generator   (band.disc_q)

Each agent awaits its inbox and sends to the other's inbox directly.
No orchestrator drives the sequence — agents block on their queue and
wake up the moment a message arrives.

All messages are also appended to a JSON log file for debugging.
"""

import asyncio
import json
from pathlib import Path

_STATE_DIR = Path("band_states")


class Band:
    def __init__(self, session_id: str):
        _STATE_DIR.mkdir(exist_ok=True)
        self.session_id = session_id
        self._log_path = _STATE_DIR / f"{session_id}.json"
        self._log: list[dict] = []

        # One queue per direction — unbounded, so sends never block
        self.gen_q: asyncio.Queue = asyncio.Queue()   # generator  → discriminator
        self.disc_q: asyncio.Queue = asyncio.Queue()  # discriminator → generator

    # ---------------------------------------------------------------- send/recv

    async def generator_send(self, msg: dict):
        """Generator posts to discriminator's inbox."""
        self._append_log("generator", msg)
        await self.gen_q.put(msg)

    async def discriminator_send(self, msg: dict):
        """Discriminator posts to generator's inbox."""
        self._append_log("discriminator", msg)
        await self.disc_q.put(msg)

    async def generator_recv(self) -> dict:
        """Generator blocks until discriminator sends something."""
        return await self.disc_q.get()

    async def discriminator_recv(self) -> dict:
        """Discriminator blocks until generator sends something."""
        return await self.gen_q.get()

    # --------------------------------------------------------------- logging

    def _append_log(self, sender: str, msg: dict):
        self._log.append({"from": sender, "msg": msg})
        self._log_path.write_text(
            json.dumps({"session_id": self.session_id, "messages": self._log}, indent=2)
        )

    @property
    def log_path(self) -> str:
        return str(self._log_path)
