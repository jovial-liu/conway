# Changelog

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
