from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import time
import uuid
from .actions import validate_action
from .control import EmergencyStop, SessionControl


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
                 max_errors: int = 5, context_chars: int = 16000, max_seconds: float = 0, quiet: bool = False) -> None:
        self.brain, self.executor, self.memory = brain, executor, memory
        self.computer = executor.computer
        self.dry_run, self.interval, self.max_steps = dry_run, max(0, interval), max_steps
        self.screenshot_keep = screenshot_keep
        self.memory_compaction_bytes = min(memory_compaction_bytes, max(1000, context_chars // 2))
        self.max_errors, self.context_chars, self.max_seconds, self.quiet = max_errors, context_chars, max_seconds, quiet

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
        state.status, state.dry_run, state.stop_reason = 'running', self.dry_run, None
        self.control = SessionControl(self.memory.root, state.session_id)
        self.executor.control = self.control
        if recovery:
            self.memory.journal({'event': 'recovery', 'pending_action': recovery,
                                 'result': 'Previous action outcome is uncertain. Re-observe; never automatically replay.'})
            state.last_result = 'Previous action outcome uncertain; a fresh observation is required.'
        self.memory.save_state(state)
        completed, errors = 0, 0
        deadline = time.monotonic() + self.max_seconds if self.max_seconds else None

        def paused() -> bool:
            was_paused = False
            while self.control.command() == 'pause':
                was_paused = True
                state.status = 'paused'
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
                try:
                    constitution = self.memory.constitution()
                    observation = self.computer.observe(state.cycle)
                    state.last_observation = observation.summary()
                    decision = self.brain.decide(constitution, self.memory.recall(self.context_chars), observation)
                    self.control.check()
                    if paused():
                        continue  # Discard a pre-pause decision; its screenshot is stale.
                    if deadline and time.monotonic() >= deadline:
                        raise EmergencyStop('Time budget reached during inference')
                    action = validate_action(decision.action, observation.width, observation.height)
                    if action['type'] not in self.executor.enabled_actions():
                        raise PermissionError(f"Tool disabled: {action['type']}")
                    record = action_record(action)
                    state.last_action = json.dumps(record, ensure_ascii=False)
                    state.last_rationale = decision.rationale
                    if self.dry_run:
                        result = f"planned only: {action['type']}; no action executed"
                    else:
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
                    errors = 0
                    if not self.dry_run:
                        self.memory.append_memory(decision.memory_note)
                        self._compact_memory(constitution, state)
                    if not self.quiet:
                        print(json.dumps({'cycle': state.cycle, 'action': action['type'], 'mode': 'plan' if self.dry_run else 'execute'}), flush=True)
                    if action['type'] == 'finish':
                        state.status = 'planned' if self.dry_run else ('error' if action['args']['outcome'] == 'failed' else 'completed')
                        state.stop_reason = action['args']['reason'] or 'Model ended the session'
                        break
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
                        state.status, state.stop_reason = 'error', 'Consecutive error limit reached'
                        break
                    self.control.sleep(min(10, 0.5 * 2**(errors - 1)))
                finally:
                    try:
                        self.computer.prune_screenshots(keep=self.screenshot_keep)
                    except OSError:
                        pass  # Screenshot cleanup must not replace the primary outcome.
                    self.memory.save_state(state)
                self.control.sleep(self.interval)
        except EmergencyStop as exc:
            state.status, state.stop_reason = 'stopped', str(exc)
        except KeyboardInterrupt:
            state.status, state.stop_reason = 'stopped', 'Keyboard interrupt'
            raise
        finally:
            if state.status in {'running', 'paused'}:
                state.status = 'stopped'
                state.stop_reason = state.stop_reason or 'Step budget reached'
            state.pid = None
            self.memory.save_state(state)
        return state.status
