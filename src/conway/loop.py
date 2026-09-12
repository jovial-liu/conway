from __future__ import annotations

import time

from .brain import Brain
from .computer import Computer
from .memory import FileMemory


class ConwayLoop:
    def __init__(
        self,
        brain: Brain,
        computer: Computer,
        memory: FileMemory,
        *,
        dry_run: bool = False,
        interval: float = 0.5,
        max_steps: int = 0,
    ) -> None:
        self.brain = brain
        self.computer = computer
        self.memory = memory
        self.dry_run = dry_run
        self.interval = max(0.0, interval)
        self.max_steps = max_steps

    def run(self) -> None:
        state = self.memory.load_state()
        state.status = "running"
        self.memory.save_state(state)

        completed = 0
        try:
            while self.max_steps <= 0 or completed < self.max_steps:
                state.cycle += 1
                observation = self.computer.observe(state.cycle)
                constitution = self.memory.constitution()
                recalled = self.memory.recall()

                try:
                    decision = self.brain.decide(constitution, recalled, observation)
                    action = decision.action
                    if self.dry_run:
                        result = f"dry-run: would execute {action}"
                    else:
                        result = self.computer.execute(action)

                    state.last_action = str(action)
                    state.last_result = result
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
                except Exception as exc:
                    state.last_result = f"error: {type(exc).__name__}: {exc}"
                    self.memory.journal(
                        {
                            "cycle": state.cycle,
                            "observation": observation.summary(),
                            "error": state.last_result,
                        }
                    )
                    time.sleep(max(1.0, self.interval))

                self.memory.save_state(state)
                completed += 1
                if self.interval:
                    time.sleep(self.interval)
        finally:
            state.status = "stopped"
            self.memory.save_state(state)
