from __future__ import annotations

from datetime import datetime, timezone
import time

from .actions import validate_action
from .brain import Brain
from .memory import FileMemory
from .tools import ToolExecutor


class ConwayLoop:
    def __init__(
        self,
        brain: Brain,
        executor: ToolExecutor,
        memory: FileMemory,
        *,
        dry_run: bool = False,
        interval: float = 0.5,
        max_steps: int = 0,
        screenshot_keep: int = 30,
        memory_compaction_bytes: int = 64_000,
    ) -> None:
        self.brain = brain
        self.executor = executor
        self.computer = executor.computer
        self.memory = memory
        self.dry_run = dry_run
        self.interval = max(0.0, interval)
        self.max_steps = max_steps
        self.screenshot_keep = max(1, screenshot_keep)
        self.memory_compaction_bytes = max(8000, memory_compaction_bytes)

    def _compact_memory(self, constitution: str, state) -> None:
        if not self.memory.needs_compaction(self.memory_compaction_bytes):
            return
        try:
            summary = self.brain.compact_memory(constitution, self.memory.compaction_source())
            if summary:
                before = self.memory.memory_bytes()
                self.memory.replace_memory(summary)
                after = self.memory.memory_bytes()
                state.compactions += 1
                self.memory.journal(
                    {
                        "cycle": state.cycle,
                        "event": "memory_compaction",
                        "before_bytes": before,
                        "after_bytes": after,
                    }
                )
                return
        except Exception as exc:
            self.memory.journal(
                {
                    "cycle": state.cycle,
                    "event": "memory_compaction_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        self.memory.compact_if_needed(
            max_bytes=max(self.memory_compaction_bytes * 4, 256_000),
            keep_lines=500,
        )

    def run(self) -> None:
        state = self.memory.load_state()
        state.status = "running"
        if not state.started_at:
            state.started_at = datetime.now(timezone.utc).isoformat()
        self.memory.save_state(state)

        completed = 0
        consecutive_errors = 0
        try:
            while self.max_steps <= 0 or completed < self.max_steps:
                state.cycle += 1
                observation = None
                constitution = self.memory.constitution()
                try:
                    observation = self.computer.observe(state.cycle)
                    state.last_observation = observation.summary()
                    recalled = self.memory.recall()
                    decision = self.brain.decide(constitution, recalled, observation)
                    action = validate_action(decision.action, observation.width, observation.height)
                    if self.dry_run:
                        result = f"observation-only: would execute {action}"
                    else:
                        result = self.executor.execute(action, observation)

                    consecutive_errors = 0
                    state.last_action = str(action)
                    state.last_result = result
                    state.last_rationale = decision.rationale
                    self.memory.append_memory(decision.memory_note)
                    self.memory.journal(
                        {
                            "cycle": state.cycle,
                            "observation": observation.summary(),
                            "action": action,
                            "rationale": decision.rationale,
                            "result": result,
                            "dry_run": self.dry_run,
                        }
                    )
                    self._compact_memory(constitution, state)
                except Exception as exc:
                    consecutive_errors += 1
                    state.error_count += 1
                    state.last_result = f"error: {type(exc).__name__}: {exc}"
                    event = {
                        "cycle": state.cycle,
                        "error": state.last_result,
                    }
                    if observation is not None:
                        event["observation"] = observation.summary()
                    self.memory.journal(event)
                    time.sleep(min(30.0, max(1.0, self.interval or 1.0) * consecutive_errors))

                self.computer.prune_screenshots(keep=self.screenshot_keep)
                self.memory.save_state(state)
                completed += 1
                if self.interval:
                    time.sleep(self.interval)
        finally:
            state.status = "stopped"
            state.pid = None
            self.memory.save_state(state)
