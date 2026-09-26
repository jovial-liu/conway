# Validation scope and acceptance

This release separates implemented behavior from deployment claims.

Version 0.8.0 adds official-SDK MCP stdio and Agent Skills support plus reviewed visual-policy data export. Tests run a real MCP subprocess, verify stateful calls, allowlists/schema validation, environment forwarding, timeouts, cancellation and cleanup; a scripted autonomous policy reads a skill and invokes tools across a subgoal boundary. Dataset tests exercise prompt alignment, hashes, evidence requirements, split leakage and the training script's no-model validation path. These use synthetic fixtures, not actual VLM decisions or real training demonstrations. No GPU training step or third-party production MCP service was tested.

Version 0.7.0 adds the [continuous autonomous loop](AUTONOMOUS.md). New lifecycle tests verify continued operation after subgoal completion/failure, transient recovery, repeated-action suppression, interruptible backoff, no stdin dependency and cooperative SIGTERM cleanup. They are synthetic protocol/lifecycle evidence, not real-model long-duration task-performance results.

Version 0.6.0 adds `preflight`, `vision-check` and session tasks. Follow [device acceptance](ACCEPTANCE.md) to collect evidence on the target machine. The new tests cover seeded image generation, hit/miss/error accounting, generic/OpenCUA image protocol fixtures, diagnostic cleanup, timed capture, private metadata omission, exclusive report writes and task lifecycle. A pixel-reading HTTP fixture verifies the scoring pipeline; it is not a real VLM and its scores are not model benchmark results.

## Automated coverage

The new regression suite covers strict config/action parsing, malformed model responses, retries, model-ID negotiation, local server command construction, occupied ports, atomic state and recovery, journal repair, context bounds, GUI scaling/dispatch fixtures, clipboard failures, UI-provider timeout fallback, current-user file/Shell tools, stop/error/time budgets, export and source publication selection.

A local HTTP fixture also exercises `/v1/models`, multimodal request encoding, action parsing, real temporary-file I/O, subsequent context and terminal state in one pipeline. A subprocess smoke test runs `python -m conway ... start --mock`. Neither test contains an actual VLM or native desktop.

Reproduce:

```sh
python -m pip install -e '.[dev,mcp]'
python -m pytest -q
python -m coverage run -m pytest -q
python -m coverage report
python -m compileall -q src
python -m pip wheel . --no-deps -w dist
```

The existing CI matrix runs Ubuntu, Windows and macOS with Python 3.11. Check the specific commit's CI results, not a prior green workflow. CI success does not imply a real desktop, GPU or model was tested. Local v0.8 release testing used Python 3.12 on CPU Linux.

## Not yet verified end to end

| Area | Implemented | Remaining acceptance |
|---|---|---|
| Generic VLM | Image request and JSON action protocol | Load the chosen real quantized weights and measure useful GUI-task behavior |
| OpenCUA | Strict literal adapter and smart-resize mapping | Run actual model with matching processor settings; the official benchmark harness differs |
| PyAutoGUI | Primary-display inputs and screenshots | Permission-granted Mac/Windows/X11 sessions, application behavior and diverse DPI configurations |
| Native UI tree | Optional AX/UIA/AT-SPI read-only adapters | Actual native-provider checks, labels, permissions and timeout behavior on target machines |
| Hardware resolver | Free-memory estimates, CPU/Metal/CUDA candidates | Actual latency, memory peaks, driver/build compatibility and image-token overhead |
| Packaging | Standard Python wheel/source distribution | Signed desktop installers, auto-update and OS-specific notarization are not implemented |
| Wayland/multiple displays | Explicit limitations | Native Wayland control and complete multi-monitor coordinate handling are not implemented |
| Policy training | Reviewed visual datasets, integrity checks and experimental offline LoRA recipe | No GPU training run, trained Conway weights, measured learning gain or automatic promotion |
| Ecosystem | Official-SDK stdio fixture and local Agent Skills loader | Third-party service compatibility, remote HTTP/OAuth and broader skill collections |

## Suggested device acceptance

1. Install in a fresh virtual environment. Run `init`, `doctor`, `config --check` and three mock cycles.
2. Run `probe` against the chosen real VLM. Record model repository/revision, quantization, inference engine version, processor pixel limits and memory peaks. A protocol-only pass is insufficient to claim vision accuracy.
3. On a disposable desktop/account, establish a narrow goal in the constitution. Start with observation-only mode and inspect proposed coordinates.
4. Run a small `--execute --max-steps` session involving a disposable text editor file. Verify the external result manually, then test pause, resume, stop and the corner failsafe.
5. Interrupt one run, inspect the intent/result journal and restart. Confirm no automatic replay. Test Unicode input and DPI mappings on the actual monitor.

This repository does not attach fabricated benchmark scores or success percentages to those unperformed checks.
