---
license: mit
tags:
  - computer-use
  - agent
  - code
---
# Conway

**A local autonomous computer-use harness. No chat box. No database. A replaceable VLM.**

[中文使用说明](README.zh-CN.md) · [Installation](docs/INSTALL.md) · [Configuration](docs/CONFIGURATION.md) · [Validation and limitations](docs/VALIDATION.md) · [Changelog](CHANGELOG.md)

Conway loads a constitution, observes the desktop, asks a VLM for one next action, dispatches it, records the outcome, and observes again. The model can choose direct file/system tools or screenshot-driven GUI interaction. GitHub hosts development; Hugging Face distributes source and references existing model weights. This repository is **not a newly trained model** and does not run a hosted agent.

**Status: v0.5.0 engineering release candidate.** Offline protocol/lifecycle tests are not evidence of real-model GUI accuracy. Native accessibility adapters remain experimental. Read the validation matrix before unattended use.

## Start

Requires Python 3.11 or newer. In a virtual environment, from this repository:

```sh
python -m pip install -e .
conway init
conway doctor
conway start --mock --max-steps 3
```

The mock command is genuinely offline: synthetic PNG, no VLM download, no desktop permissions, and no computer actions. Edit the `constitution.md` path printed by `conway init` to establish the ongoing goal and scope.

For real local inference, install a compatible `llama-server` first. Then:

```sh
conway probe
conway start --max-steps 3
conway start --execute
```

`probe` sends a synthetic PNG and checks the model protocol; it never touches your desktop. `start` without `--execute` captures real screenshots and plans but does not dispatch actions. `--execute` enables the configured current-user tools for the process; there is no per-action approval dialog.

For an existing multimodal server, auto-discover its served model ID:

```sh
conway probe --endpoint http://127.0.0.1:8000/v1
conway start --endpoint http://127.0.0.1:8000/v1 --execute
```

With multiple served models, pass `--model` using a returned `/v1/models` ID. An API's model alias need not equal its Hugging Face repository ID.

## Runtime design

```text
constitution.md + rolling file context
                  |
      observe -> decide -> validate -> record intent -> act
         ^                                        |
         +----------- next observation <--- record result

Brain: generic JSON VLM | OpenCUA literal action adapter
Tools: GUI | shell | read/write/list files | open URL | finish
State: Markdown + atomic JSON + append-only JSONL
Control: pause / resume / stop + step/time/error budgets
```

Conway does not evaluate OpenCUA-generated Python. Its AST parser translates one allowlisted literal call into the same normalized action schema used by the generic VLM. Smart-resized image coordinates are mapped back to screenshot pixels, then to the host input coordinate space. Server-specific processor limits must match the configured OpenCUA pixel limits.

## Operating controls

```sh
conway status
conway pause
conway resume
conway stop
conway config --check
conway models
conway start --execute --max-steps 100 --max-seconds 600
```

Controls are cooperative and scoped to one session. A stop arriving during inference prevents its returned action from dispatching; an in-flight native operation still needs to return. `Ctrl+C` and the PyAutoGUI corner failsafe remain available. `--max-seconds` is not an OS-enforced hard deadline.

An intent record is written **before** a side effect. If the process crashes or dispatch raises after partial work, Conway records an uncertain outcome and does not automatically replay the action. A restarted session obtains a fresh observation. A model's `finish` declaration is recorded separately from independently verified task success.

## Models and hardware

| Profile | Reference | Execution |
|---|---|---|
| `tiny` | `mradermacher/Qwen3-VL-2B-Instruct-abliterated-GGUF` | local GGUF, Q4_K_M |
| `small` | `mradermacher/Qwen3-VL-4B-Instruct-abliterated-GGUF` | local GGUF, Q4_K_M |
| `standard` | `mradermacher/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF` | local GGUF, Q4_K_M |
| `computer-use` | `xlangai/OpenCUA-7B` | compatible external multimodal server |

The resolver uses available RAM or free memory on one NVIDIA GPU, reserves headroom, and refuses an estimated non-fitting local model. Apple Silicon uses a unified-memory budget. Memory thresholds are estimates, not measured throughput or guaranteed capacity. AMD/Intel GPU acceleration is not auto-verified; use a compatible external backend or CPU fallback. The runtime binary is not installed automatically.

A community `abliterated` label concerns refusal behavior; it is not a promise of GUI accuracy, reliability, or any specific training method. Referenced weights retain their own upstream licenses.

## GUI and direct tools

Primary-display screenshot, click, double-click, movement, drag, text, key, hotkey and scroll are implemented through PyAutoGUI. HiDPI input mapping is tested with fixtures; monitor rotation, multiple displays, and live OS sessions require device validation. Native Wayland input is not implemented.

Optional read-only accessibility snapshots add foreground-window labels through macOS AX, Windows UIA or Linux AT-SPI. They are disabled by default, run in a time-limited subprocess, and fall back to screenshots on failure. These adapters do not yet provide element-ID actions.

Direct tools provide bounded file reads, atomic text replacement, append, directory listing, browser opening and current-user shell commands. Shell output is bounded and command descendants are cleaned up on timeout. Tool configuration and `--gui-only` are routing preferences, **not a sandbox**: a GUI-capable agent may still operate a terminal under your account.

## File state

```text
CONWAY_HOME/
  constitution.md         owner-defined goals; existing file is preserved
  config.yaml             validated configuration; no API keys
  memory.md               rolling summary
  memory.previous.md      previous summary excerpt
  state.json              current state and pending action
  state.previous.json     recoverable prior state
  control.json            session-scoped control request
  instance.lock           single-instance OS file lock
  journal/*.jsonl          intents, results, errors and recovery events
  screenshots/*.png       bounded recent screenshots
  llama-runtime.log       local inference startup diagnostics
```

Use `CONWAY_HOME`, or put `--home PATH` **before** the subcommand. Memory compaction is bounded by both the configured threshold and the context budget; fallback truncation is explicitly marked. No online weight updates or continual training occur. Journals are retained locally and rotated by size, not silently uploaded.

```sh
conway export --output ./trajectory.jsonl
```

Export excludes dry runs by default, omits image bytes and leaves task-success labels unknown. Review file contents, action text and logs for private information before sharing.

## Development and publication

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python -m compileall -q src
python -m pip wheel . --no-deps -w dist
```

The test suite exercises parsing, HTTP failures, lifecycle, crash recovery, file/shell I/O, coordinate conversion, UI fallback and a local HTTP fixture-to-tools pipeline. The fixture is not a real model. Existing GitHub CI runs on Linux, Windows and macOS.

HF publication uses a **source snapshot**, not a force-pushed Git history. Tracked source, tests, docs and examples are selected by `scripts/publish_hf.py`; experiment data, runtime state, secrets, binaries and model weights are excluded. `_source_commit.json` records the GitHub commit and per-file hashes and is read back from the HF publication commit. GitHub's existing experiment files and history remain untouched.

## Boundary

Conway runs with the current account's OS permissions. It does not implement OS-permission bypass, privilege escalation, stealth startup or network self-propagation. Do not assume a prompt or config flag provides isolation. Start on a disposable desktop/account and make backups before granting write-capable tools to an autonomous model.

## References

- [OpenCUA-7B model card](https://huggingface.co/xlangai/OpenCUA-7B)
- [llama.cpp multimodal documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
- [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [PyAutoGUI documentation](https://pyautogui.readthedocs.io/en/latest/)
- [Hugging Face upload documentation](https://huggingface.co/docs/huggingface_hub/guides/upload)

MIT for Conway's code; model licenses are separate.
