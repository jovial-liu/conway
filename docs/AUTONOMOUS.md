# Continuous autonomous loop

The primary entry point is `conway run`. The agent reads its constitution and file
memory, observes, chooses one action, records intent, acts, observes the result,
and continues. It does not expose a conversation window, input prompt or chat
server. It never reads stdin. The model selects work within the constitution;
the loop does not require a new user message between tasks.

## Start

After installing Conway and configuring the model, set ongoing objectives and
scope in the constitution path printed by `conway init`. An example is in
`examples/constitution.autonomous.md`. Existing constitutions are not overwritten.

```sh
conway config --check
conway preflight --desktop
conway run
```

`run` executes enabled GUI/file/Shell/URL tools by default. `run --observe` uses
the same continuous lifecycle but never dispatches actions. `run --gui-only`
disables direct file/Shell/URL tools. `run --mock` uses a synthetic desktop and
wait-only brain without connecting a model or executing real computer actions.
`--max-steps` and `--max-seconds` optionally bound execution; zero means no such
budget. `--quiet` removes startup/cycle console output, leaving state and journal.

The process remains in the terminal or process manager that launched it. It does
not open a window, detach itself, install a boot service or restart itself after
stop. Use another terminal for `status`, `pause`, `resume` and `stop`. These are
lifecycle controls, not task-conversation messages. The existing `start` command
is retained for bounded single-session acceptance, with its old task flags and
finish semantics; those flags are intentionally absent from `run`.

## Subgoals and pacing

In continuous mode, `finish`, OpenCUA `DONE` and OpenCUA `FAIL` report the current
subgoal's outcome. The process records `goal_report`, increments `goal_reports`,
stores `last_goal_report` with `verified: false`, and then re-observes. A model
declaration is not independent verification. Dry-run goal reports remain tagged
as such and never add fictional completed work to durable memory.

The continuous-operation directive is supplied on every cycle to both supported
brain adapters. It asks the model to choose useful work itself, inspect results,
preserve unfinished work, and wait when nothing useful remains. Earlier finishes
do not become instructions to stop. No separate worker agents are spawned.

On unchanged screenshots, repeated wait/finish decisions use idle delays of
2, 4, 8…60 seconds by default. A changed screenshot/display/window or active work
resets idle escalation. The model's explicit wait can lengthen an idle delay up
to the action schema's bound. Screenshots are sampled when the next cycle begins;
this is periodic reassessment, not an instantaneous desktop-event subscription.

Before the `max_errors` threshold, errors receive short bounded backoff. At the
threshold, continuous mode enters longer recovery: 5, 10, 20…300 seconds by
default, then retries from a fresh observation. Successful decisions reset the
consecutive error count. This recovers transient inference or read-only failures;
it does not reinstall drivers or restart a crashed local model server. Startup
configuration/runtime failures remain explicit startup failures.

## Repeated actions

The default limit allows three identical side-effect attempts while screenshot
bytes, display geometry and active window/application remain unchanged. A further
repeat is not dispatched; `action_suppressed` is recorded and made available to
the next model decision so it can change approach or wait. Wait/finish decisions
do not clear this guard; a different side-effect action or changed observation
does. Read-only file/list operations are not counted as side effects.

This is a conservative repetition heuristic, not proof of progress. Animated
pixels may reset it, and legitimate repeated commands without visible change may
be suppressed. It does not detect every multi-action cycle. Configure
`repeat_action_limit` if the intended workflow requires more repetitions.

## Stop and recovery semantics

State exposes `loop_mode`, live `idle`/`recovering` phases and `next_wake_at`.
Pause, resume and stop are checked during waits, not only after the full delay.
Resume discards the old waiting schedule and obtains a fresh observation. During
inference, control commands take effect at cooperative checkpoints; HTTP timeouts
still bound in-flight requests. A session time budget clips scheduled waits.

Explicit stop, Ctrl+C, mouse failsafe, cooperative SIGTERM and an uncertain
side-effect error end execution. A partial write/click/Shell operation is not
blindly repeated by the recovery scheduler. The pending intent remains on disk,
and a later user-started run records uncertainty and re-observes as before.
SIGTERM restores its previous handler after cleanup; forced process termination
on Windows or SIGKILL cannot promise cleanup. There is no watchdog respawn.

## Validation scope

Regression tests cover multiple subgoal boundaries, generic model HTTP protocol
fixtures, real temporary-file actions, transient outages, failed reads, partial
writes, idle pacing, repeated-action suppression, pause/resume/stop during long
cooldowns, deadlines, and a stdin-free subprocess with cooperative SIGTERM on
POSIX. They do not measure a real VLM's task competence or native desktop success.
Use `docs/ACCEPTANCE.md` on the target device for that evidence. Long-term journals
remain local files and require owner-managed archiving as documented previously.
