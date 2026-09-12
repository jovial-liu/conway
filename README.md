# Conway

**Conway is a local-first, headless autonomous computer-use agent harness.**

Conway continuously observes the current computer state, reasons with a pluggable vision-language model, chooses the next action, executes it through a cross-platform computer adapter, records a lightweight file-based journal, and repeats — without requiring a chat interface or a human prompt for every step.

> Status: early v0.1 scaffold. APIs and model manifests will change.

## What v0.1 contains

- **Headless loop** — observe → decide → act → record → repeat.
- **VLM-first perception** — the current screenshot is sent to a multimodal model each cycle.
- **Computer use** — mouse, keyboard, hotkeys, scrolling, dragging, and screenshots.
- **Cross-platform base layer** — PyAutoGUI-backed input/screenshot adapter for macOS, Windows, and Linux desktop sessions.
- **Hardware detection** — RAM, Apple unified-memory/Metal, NVIDIA VRAM/CUDA, and CPU fallback.
- **Automatic model profile selection** — local 4B/8B multimodal GGUF profiles selected from detected effective memory.
- **OpenAI-compatible model adapter** — works with local llama.cpp-style and compatible multimodal servers.
- **File-based context** — `constitution.md`, `memory.md`, `state.json`, and JSONL journals; no SQL/vector DB.
- **No per-action Conway approval dialogs** once GUI execution is explicitly enabled at startup.
- **OS-native security boundary** — Conway does not bypass OS permissions, UAC, TCC, sudo, sandboxing, or other platform security controls.

## Architecture

```text
constitution.md
      │
      ▼
Autonomous Loop ─────── File Context / Journal
      │
  ┌───┴──────────┐
  ▼              ▼
Brain         Computer
(VLM)        (OS adapter)
  │              │
  ▼              ▼
Model       Screen / Mouse /
Backend     Keyboard / GUI
```

## Install

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

For automatic local model startup, install a current `llama.cpp` build so either `llama-server` or `llama` is available on `PATH`.

## Run

Inspect the machine and see which profile Conway would select:

```bash
conway doctor
```

Observation-only mode (the model can inspect and plan, but Conway does not execute GUI actions):

```bash
conway start
```

Enable GUI execution for the running Conway process:

```bash
conway start --execute
```

Run a bounded session:

```bash
conway start --execute --max-steps 100
```

Exercise the loop without downloading/loading a VLM:

```bash
conway start --mock --max-steps 5
```

Connect to an already-running OpenAI-compatible multimodal server:

```bash
conway start --endpoint http://127.0.0.1:8000/v1 --model your-model --execute
```

Your OS may require normal user-granted permissions such as Accessibility and Screen Recording on macOS. Conway does not attempt to bypass those controls.

## Model profiles

The current manifest lives at `src/conway/models.yaml`.

| Profile | Model | Runtime |
| --- | --- | --- |
| `small` | `mradermacher/Qwen3-VL-4B-Instruct-abliterated-GGUF` (`Q4_K_M`) | auto-start via llama.cpp |
| `standard` | `mradermacher/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF` (`Q4_K_M`) | auto-start via llama.cpp |
| `computer-use` | `xlangai/OpenCUA-7B` | external compatible server |

The first two are community-modified low-refusal variants, not official Qwen safety-tuned releases. The model layer is deliberately replaceable; Conway is the harness, not a particular checkpoint.

## Local state

Conway stores runtime state under the platform-standard user data directory:

```text
conway/
├── constitution.md
├── memory.md
├── state.json
├── llama-runtime.log
├── screenshots/
└── journal/
    └── YYYY-MM-DD.jsonl
```

No SQL database or vector database is required for v0.1.

## Current limitations

- The v0.1 computer adapter uses screenshot + coordinate interaction; native accessibility-tree adapters are the next major computer-use improvement.
- Wayland environments may restrict screenshot/input automation depending on compositor policy.
- Model selection is hardware-aware but intentionally conservative; users can override it with `--profile` or provide `--endpoint`.
- This repository is an early scaffold and has not yet been packaged as a signed desktop application.

## Security boundary

Conway operates with permissions already granted to the current user account. It deliberately does **not** implement privilege escalation, permission bypass, credential harvesting, stealth/persistence mechanisms, or security-control evasion.

## License

MIT
