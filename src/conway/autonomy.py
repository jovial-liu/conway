"""Pacing and repeated-action detection for the continuous, non-interactive loop."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json


CONTINUOUS_DIRECTIVE = '''
## Continuous autonomous operation
You run continuously from this constitution, file memory and fresh observations.
There is no conversation window, incoming chat queue or user turn to wait for.
Independently select useful work within the constitution, perform one action,
observe the result, and revise your plan. Preserve unfinished work in memory.
When one subgoal is complete, choose the next useful subgoal. finish/DONE/FAIL
reports only a subgoal boundary; it does not stop this process. Never interpret
an earlier finish as an instruction to stop. Do not repeatedly announce completion.
If no useful work is available, choose wait and reassess after the idle period.
Do not manufacture work, repeat ineffective actions, or claim a dispatched action
is verified success. A suppressed repeat requires a different approach or waiting
for changed conditions. pause, stop and operating-system permissions remain binding.
'''


class RepeatedAction(RuntimeError):
    """A scheduling event, not a failed or dispatched tool action."""


@dataclass
class AutonomyPolicy:
    idle_initial_seconds: float = 2.0
    idle_max_seconds: float = 60.0
    recovery_initial_seconds: float = 5.0
    recovery_max_seconds: float = 300.0
    repeat_action_limit: int = 3
    idle_streak: int = 0
    observation_key: str | None = None
    action_key: str | None = None
    action_count: int = 0

    def observe(self, observation) -> None:
        with observation.screenshot_path.open('rb') as image:
            pixels = hashlib.file_digest(image, 'sha256').hexdigest()
        key = json.dumps([pixels, observation.width, observation.height,
                          observation.input_width, observation.input_height,
                          observation.active_app, observation.active_window], ensure_ascii=False)
        if key != self.observation_key:
            self.idle_streak = 0
            self.action_key, self.action_count = None, 0
        self.observation_key = key

    def check_repeat(self, action: dict) -> None:
        key = hashlib.sha256(json.dumps(action, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key != self.action_key:
            self.action_key, self.action_count = key, 0
        if self.action_count >= self.repeat_action_limit:
            raise RepeatedAction('Repeated identical action on an unchanged desktop suppressed; observe results and choose another approach or wait.')
        self.action_count += 1

    def idle_delay(self) -> float:
        self.idle_streak += 1
        return min(self.idle_max_seconds, self.idle_initial_seconds * 2 ** min(self.idle_streak - 1, 20))

    def recovery_delay(self, extra_errors: int) -> float:
        return min(self.recovery_max_seconds, self.recovery_initial_seconds * 2 ** min(max(0, extra_errors), 20))
