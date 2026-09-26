# Changelog

## 0.8.0

- Add optional official MCP SDK 2.2 stdio clients with persistent sessions, paginated discovery, explicit tool/environment allowlists, schema validation, bounded output and cancellation cleanup. Uncertain calls are never replayed.
- Add local Agent Skills metadata discovery and paginated instruction/resource loading, with a reserved extension-result context budget and no script auto-execution.
- Add opt-in visual episodes sharing the runtime prompt builder, independent-review requirements, integrity-checked multimodal training export and episode/image split checks. Mock and dry-run samples cannot become training demonstrations.
- Add an experimental TRL/PEFT LoRA entry point and a no-model dataset-validation mode. GPU training and Conway-trained checkpoints remain unperformed/unreleased.
- Document a model-plus-harness architecture, official ecosystem references and measurable digital-life research milestones; update English/Chinese positioning.
- Extend cross-platform CI to test the real MCP subprocess protocol and reviewed-data workflow.

## 0.7.0

- Make `conway run` the continuous autonomous entry point: executes enabled tools from the constitution, never reads stdin, and opens no chat UI. `--observe` supports continuous planning only; `start` retains single-session compatibility.
- Treat `finish`/OpenCUA `DONE`/`FAIL` as subgoal reports in continuous mode, preserving explicit unverified provenance and continuing from fresh observations.
- Add bounded exponential idle and recovery delays, live idle/recovering state and next-wake metadata; recover from pre-dispatch/readonly failures beyond the former session error limit.
- Suppress repeated identical side-effect actions on unchanged observations and feed the suppression back into context.
- Keep pause/resume/stop and time budgets responsive during waits, preserve uncertain-action halts, and handle cooperative SIGTERM with cleanup and no respawn.
- Stop treating read-only failures as uncertain side effects; ensure stop signals during memory compaction are not swallowed.
- Add unattended multi-goal, stdin-free process, retry, pacing, repeat-guard and lifecycle tests plus continuous-operation documentation.

## 0.6.0

- Add `preflight` for constitution/recovery, hardware, runtime/service and GUI-dependency checks without downloading weights; optional timed desktop capture removes its temporary screenshot and omits private window/UI labels.
- Add `vision-check` with seeded synthetic color targets, independent coordinate scoring, per-case image hashes, hit rate, latency and explicit non-desktop scope. Returned actions are never executed.
- Support both generic JSON and OpenCUA adapters through the existing model session, with cleanup on failures.
- Add `start --task` / `--task-file`; validate before runtime startup, retain the task in state/journal and supply it each cycle without rewriting the constitution or automatically reusing the assignment.
- Add exclusive local JSON report output, meaningful diagnostic exit codes, device-acceptance documentation and regression tests.
- Preserve v0.5.0 config/state compatibility, file-only memory, source snapshot publication and existing experiment history.

## 0.5.0

- Validate configuration types, duplicate keys and numeric ranges; fix quoted `false` being truthy.
- Strict model-list readiness and served-alias discovery; stop accepting HTTP 404/401 as a ready model.
- Use only an explicit `llama-server` binary; detect port conflicts, configure context and clean owned processes.
- Reject ambiguous OpenCUA programs, non-literal arguments and duplicate model JSON; support terminal failure distinctly.
- Add session-scoped pause/resume/stop, real offline smoke mode, synthetic-image protocol probe and bounded sessions.
- Preserve pending side-effect intents and surface uncertainty after failures; never automatically replay interrupted actions.
- Fix failsafe retry behavior, stale display/window checks and Unicode clipboard failure handling.
- Add atomic state/summary/file replacement, recoverable previous state, partial-journal repair and bounded rolling memory.
- Bound Shell output while draining it and clean command descendants on timeout; keep current-user execution semantics.
- Add optional read-only native UI snapshots with subprocess timeouts and screenshot fallback.
- Add export with dry-run exclusion and unknown task-success labels.
- Add extensive regression/integration tests, install/config/acceptance docs and Chinese instructions.
- Standardize wheel/source packaging without bundling experiment datasets or model weights.
- Add source-snapshot publication with a commit/hash manifest, avoiding oversized experiment files and force-pushed history.

## 0.4.0 and earlier

The existing implementation introduced autonomous looping, screenshot input, GUI and direct tools,
OpenCUA support, file memory, hardware profiles, config and the GitHub/Hugging Face publication setup.
Existing user experiment additions and repository history are preserved.
