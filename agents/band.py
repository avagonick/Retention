"""
Band — shared message bus between generator and discriminator agents.

One Band instance per generation session (keyed by session_id). Each agent
posts messages to the band and reads the other agent's latest output from it.
State is persisted to a JSON file so you can inspect/debug any iteration.
"""

import json
from pathlib import Path
from typing import Any

_STATE_DIR = Path("band_states")


class Band:
    def __init__(self, session_id: str):
        _STATE_DIR.mkdir(exist_ok=True)
        self.session_id = session_id
        self._path = _STATE_DIR / f"{session_id}.json"
        self._state = self._load()

    # ------------------------------------------------------------------ I/O

    def _load(self) -> dict:
        if self._path.exists():
            return json.loads(self._path.read_text())
        return {
            "session_id": self.session_id,
            "iteration": 0,
            "messages": [],
            "current_video": None,
            "done": False,
            "final_video": None,
        }

    def _save(self):
        self._path.write_text(json.dumps(self._state, indent=2))

    # ---------------------------------------------------- message passing

    def post(self, role: str, content: dict):
        """Agent posts a message to the band."""
        self._state["messages"].append({
            "role": role,
            "iteration": self._state["iteration"],
            "content": content,
        })
        self._save()

    def latest_from(self, role: str) -> dict | None:
        """Return the most recent message posted by `role`."""
        msgs = [m for m in self._state["messages"] if m["role"] == role]
        return msgs[-1]["content"] if msgs else None

    def history(self) -> list[dict]:
        return list(self._state["messages"])

    # --------------------------------------------------------- state helpers

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set(self, key: str, value: Any):
        self._state[key] = value
        self._save()

    def bump(self):
        """Advance the iteration counter."""
        self._state["iteration"] += 1
        self._save()

    @property
    def iteration(self) -> int:
        return self._state["iteration"]

    @property
    def done(self) -> bool:
        return self._state["done"]

    def mark_done(self, final_video: str):
        self._state["done"] = True
        self._state["final_video"] = final_video
        self._save()
