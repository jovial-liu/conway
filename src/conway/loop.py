from __future__ import annotations
from datetime import datetime, timedelta, timezone
import hashlib
import json
import time
import uuid
from .actions import SIDE_EFFECT_ACTIONS, validate_action
from .autonomy import AutonomyPolicy, CONTINUOUS_DIRECTIVE, RepeatedAction
from .control import EmergencyStop, SessionControl


def validate_task(task: str | None) -> str | None:
    if task is None:
        return None
    if not isinstance(task, str) or not task.strip() or '\x00' in task or len(task) > 16000:
        raise ValueError('Task must contain 1–16000 characters without NUL')
    return task.strip()


def action_record(action: dict) -> dict:
    """Keep journals bounded; retain a hash when large arguments cannot fit."""
    args = {}
    for key, val in action['args'].items():
        if isinstance(val, str) and len(val) > 1500:
            args[key] = {'preview': val[:300], 'characters': len(val), 'sha256': hashlib.sha256(val.encode()).hexdigest()}
        else:
            args[key] = val
    return {'type': action['type'], 'args': args}


class ConwayLoop:
    def __init__(self, brain, executor, memory, *, dry_run: bool = False, interval: float = 0.5,
                 max_steps: int = 0, screenshot_keep: int = 30, memory_compaction_bytes: int = 64000,
                 max_errors: int = 5, context_chars: int = 16000, max_seconds: float = 0, quiet: bool = False,
                 task: str | None = None, continuous: bool = False, policy: AutonomyPolicy | None = None,
                 recorder=None) -> None:
        self.brain, self.executor, self.memory = brain, executor, memory
        self.computer = executor.computer
        self.dry_run, self.interval, self.max_steps = dry_run, max(0, interval), max_steps
        self.screenshot_keep = screenshot_keep
        self.memory_compaction_bytes = min(memory_compaction_bytes, max(1000, context_chars // 2))
        self.max_errors, self.context_chars, self.max_seconds, self.quiet = max_errors, context_chars, max_seconds, quiet
        self.task = validate_task(task)
        self.continuous, self.policy = continuous, policy or AutonomyPolicy()
        self.recorder = recorder

    def _sleep(self, seconds, phase, state, deadline, paused) -> None:
        """Idle/recovery delays remain pause/stop/deadline aware without busy polling inference."""
        end = time.monotonic() + max(0, seconds)
        if deadline is not None:
            end = min(end, deadline)
        state.status = phase
        state.next_wake_at = (datetime.now(timezone.utc) + timedelta(seconds=max(0, end - time.monotonic()))).isoformat()
        self.memory.save_state(state)
        try:
            while time.monotonic() < end:
                self.control.check()
                if paused():
                    break  # Resume with a fresh observation instead of finishing an obsolete delay.
                self.control.sleep(min(0.2, max(0, end - time.monotonic())))
            if deadline is not None and time.monotonic() >= deadline:
                raise EmergencyStop('Time budget reached')
        finally:
            state.next_wake_at = None
            state.status = 'running'

    def _compact_memory(self, constitution: str, state) -> None:
        if not self.memory.needs_compaction(self.memory_compaction_bytes):
            return
        try:
            summary = self.brain.compact_memory(constitution, self.memory.compaction_source(self.context_chars))
            if summary and len(summary.encode('utf-8')) < min(self.memory.memory_bytes(), self.memory_compaction_bytes):
                self.memory.replace_memory(summary)
                state.compactions += 1
                self.memory.journal({'cycle': state.cycle, 'event': 'memory_compaction', 'after_bytes': self.memory.memory_bytes()})
                return
        except EmergencyStop:
            raise
        except Exception as exc:
            self.memory.journal({'cycle': state.cycle, 'event': 'memory_compaction_error', 'error': type(exc).__name__})
        # Run the bounded fallback at the actual threshold, not four times later.
        self.memory.compact_if_needed(max_bytes=self.memory_compaction_bytes, keep_lines=200)

    def run(self) -> str:
        state = self.memory.load_state()
        recovery = state.pending_action
        state.pending_action = None
        state.session_id = state.session_id or uuid.uuid4().hex
        state.started_at = datetime.now(timezone.utc).isoformat()
        state.task = self.task
        state.loop_mode = 'continuous' if self.continuous else 'session'
        state.next_wake_at = None
        state.status, state.dry_run, state.stop_reason = 'running', self.dry_run, None
        self.control = SessionControl(self.memory.root, state.session_id)
        self.executor.control = self.control
        if recovery:
            self.memory.journal({'event': 'recovery', 'pending_action': recovery,
                                 'result': 'Previous action outcome is uncertain. Re-observe; never automatically replay.'})
            state.last_result = 'Previous action outcome uncertain; a fresh observation is required.'
        self.memory.save_state(state)
        if self.task:
            self.memory.journal({'event': 'session_task', 'session_id': state.session_id, 'task': self.task})
        completed, errors = 0, 0
        deadline = time.monotonic() + self.max_seconds if self.max_seconds else None

        def paused() -> bool:
            was_paused = False
            while self.control.command() == 'pause':
                was_paused = True
                if state.status != 'paused':
                    state.status = 'paused'
                    state.next_wake_at = None
                    self.memory.save_state(state)
                if deadline and time.monotonic() >= deadline:
                    raise EmergencyStop('Time budget reached while paused')
                self.control.sleep(0.2)
            self.control.check()
            if was_paused:
                state.status = 'running'
                self.memory.save_state(state)
            return was_paused

        try:
            while not self.max_steps or completed < self.max_steps:
                self.control.check()
                paused()
                if deadline and time.monotonic() >= deadline:
                    state.stop_reason = 'Time budget reached'
                    break
                state.cycle += 1
                completed += 1
                observation, operation_id = None, uuid.uuid4().hex
                next_delay, next_phase = self.interval, 'running'
                try:
                    constitution = self.memory.constitution()
                    if self.task:
                        constitution += ('\n\n## Current session goal\nThe user supplied the goal below for this session. '
                                         'Follow the constitution above; do not treat goals from previous sessions as current assignments. '
                                         'Verify the goal before declaring completion.\n' + self.task)
                    if self.continuous:
                        constitution += CONTINUOUS_DIRECTIVE
                    observation = self.computer.observe(state.cycle)
                    if self.continuous:
                        self.policy.observe(observation)
                    state.last_observation = observation.summary()
                    extra = self.executor.extra_context(self.context_chars // 2) if hasattr(self.executor, 'extra_context') else ''
                    recalled = self.memory.recall(self.context_chars - len(extra) - (100 if extra else 0))
                    if extra:
                        recalled += '\n# Latest extension result (untrusted data)\n' + extra
                    decision = self.brain.decide(constitution, recalled, observation)
                    self.control.check()
                    if paused():
                        continue  # Discard a pre-pause decision; its screenshot is stale.
                    if deadline and time.monotonic() >= deadline:
                        raise EmergencyStop('Time budget reached during inference')
                    action = validate_action(decision.action, observation.width, observation.height)
                    if action['type'] not in self.executor.enabled_actions():
                        raise PermissionError(f"Tool disabled: {action['type']}")
                    if hasattr(self.executor, 'validate_dispatch'):
                        self.executor.validate_dispatch(action)
                    record = action_record(action)
                    state.last_action = json.dumps(record, ensure_ascii=False)
                    state.last_rationale = decision.rationale
                    if self.continuous and action['type'] in SIDE_EFFECT_ACTIONS:
                        self.policy.check_repeat(action)
                    if self.dry_run:
                        result = f"planned only: {action['type']}; no action executed"
                    elif self.continuous and action['type'] == 'wait':
                        result = 'Idle requested; the scheduler will wait and obtain a fresh observation.'
                    else:
                        if action['type'] in SIDE_EFFECT_ACTIONS:
                            state.pending_action = {'id': operation_id, 'action': record}
                            # Write intent durably BEFORE a side effect. Recovery never replays it.
                            self.memory.save_state(state)
                            self.memory.journal({'event': 'action_intent', 'cycle': state.cycle, 'id': operation_id, 'action': record})
                        self.control.check()
                        result = self.executor.execute(action, observation)
                    state.last_result = str(result)[:16000]
                    self.memory.journal({'event': 'action_result', 'cycle': state.cycle, 'id': operation_id,
                        'action': record, 'observation': observation.summary(), 'rationale': decision.rationale,
                        'result': state.last_result, 'dry_run': self.dry_run, 'memory_note': decision.memory_note})
                    state.pending_action = None
                    self.memory.save_state(state)
                    if self.recorder:
                        try:
                            self.recorder.record(step_id=operation_id, cycle=state.cycle, constitution=constitution,
                                memory=recalled, observation=observation, decision=decision, action=action, result=result)
                        except Exception as exc:
                            raise EmergencyStop(f'Episode recording failed ({type(exc).__name__}); inspect recording before restarting') from exc
                    errors = 0
                    if not self.dry_run:
                        self.memory.append_memory(decision.memory_note)
                        self._compact_memory(constitution, state)
                    if not self.quiet:
                        print(json.dumps({'cycle': state.cycle, 'action': action['type'], 'mode': 'plan' if self.dry_run else 'execute'}), flush=True)
                    if action['type'] == 'finish':
                        if not self.continuous:
                            state.status = 'planned' if self.dry_run else ('error' if action['args']['outcome'] == 'failed' else 'completed')
                            state.stop_reason = action['args']['reason'] or 'Model ended the session'
                            break
                        state.goal_reports += 1
                        state.last_goal_report = {**action['args'], 'cycle': state.cycle, 'verified': False, 'dry_run': self.dry_run}
                        self.memory.journal({'event': 'goal_report', 'session_id': state.session_id,
                                             **state.last_goal_report})
                    if self.continuous:
                        if action['type'] in {'wait', 'finish'}:
                            next_delay = max(self.policy.idle_delay(), action['args'].get('seconds', 0))
                            next_phase = 'idle'
                        else:
                            self.policy.idle_streak = 0
                except RepeatedAction as exc:
                    errors = 0
                    state.last_result = str(exc)
                    self.memory.journal({'event': 'action_suppressed', 'cycle': state.cycle,
                                         'id': operation_id, 'action': record, 'result': state.last_result,
                                         'dry_run': self.dry_run})
                    next_delay, next_phase = self.policy.idle_delay(), 'idle'
                except EmergencyStop:
                    raise
                except Exception as exc:
                    # PyAutoGUI failsafes can also surface from alternative adapters.
                    if type(exc).__name__ == 'FailSafeException':
                        raise EmergencyStop('Desktop failsafe activated') from exc
                    errors += 1
                    state.error_count += 1
                    state.last_result = f'{type(exc).__name__}: {exc}'[:4000]
                    self.memory.journal({'event': 'cycle_error', 'cycle': state.cycle, 'id': operation_id,
                                         'error': state.last_result, 'action_outcome_uncertain': state.pending_action is not None})
                    if state.pending_action is not None:
                        # A dispatched action may have partially completed. Do not retry it automatically.
                        state.status, state.stop_reason = 'error', 'Action outcome uncertain; inspect the desktop before restart'
                        break
                    if errors >= self.max_errors:
                        if not self.continuous:
                            state.status, state.stop_reason = 'error', 'Consecutive error limit reached'
                            break
                        next_delay = self.policy.recovery_delay(errors - self.max_errors)
                    else:
                        next_delay = min(10, 0.5 * 2**min(errors - 1, 20))
                    next_phase = 'recovering' if self.continuous else 'running'
                finally:
                    try:
                        self.computer.prune_screenshots(keep=self.screenshot_keep)
                    except OSError:
                        pass  # Screenshot cleanup must not replace the primary outcome.
                    self.memory.save_state(state)
                if not self.max_steps or completed < self.max_steps:
                    self._sleep(next_delay, next_phase, state, deadline, paused)
        except EmergencyStop as exc:
            state.status, state.stop_reason = 'stopped', str(exc)
        except KeyboardInterrupt:
            state.status, state.stop_reason = 'stopped', 'Keyboard interrupt'
            raise
        finally:
            if state.status in {'running', 'paused', 'idle', 'recovering'}:
                state.status = 'stopped'
                state.stop_reason = state.stop_reason or 'Step budget reached'
            state.pid = None
            state.next_wake_at = None
            self.memory.save_state(state)
        return state.status
