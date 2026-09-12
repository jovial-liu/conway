# Conway

**Conway is a local-first, headless autonomous computer-use agent harness.**

Conway continuously observes the current computer state, reasons with a pluggable vision-language model, chooses the next action, executes it through a cross-platform computer adapter, records a lightweight file-based journal, and repeats — without requiring a chat interface or a human prompt for every step.

> Status: early v0.1 scaffold. APIs and model manifests will change.

## Design goals

- **Headless autonomous loop** — no chat box is required.
- **VLM-first** — screenshots are a first-class observation.
- **Computer use** — mouse, keyboard, scrolling, screenshots, and app-level interaction.
- **Cross-platform** — macOS, Windows, and Linux adapters behind one interface.
- **Hardware-aware** — model/backend selection is designed to adapt to available RAM, VRAM, Apple unified memory, and CPU-only machines.
- **Model-agnostic** — Conway is the harness; the model is replaceable.
- **File-based context** — Markdown/JSONL instead of a database in v0.1.
- **OS-native security boundary** — Conway does not add per-action approval dialogs, but it does not bypass OS permissions, UAC, TCC, sudo, sandboxing, or other platform security controls.

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
Backend     Keyboard / UI
```

## Install

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Optional local inference integrations will be added behind adapters. The initial scaffold includes a mock brain so the loop can be exercised without downloading a model.

## Run

```bash
conway doctor
conway start --dry-run
```

`--dry-run` observes and plans without executing computer actions.

To allow GUI actions:

```bash
conway start
```

Your OS may require normal user-granted permissions such as Accessibility / Screen Recording on macOS. Conway does not attempt to bypass those controls.

## Local state

Conway stores lightweight runtime state under the platform-standard user data directory:

```text
conway/
├── constitution.md
├── memory.md
├── state.json
└── journal/
    └── YYYY-MM-DD.jsonl
```

No SQL database or vector database is required for v0.1.

## Model strategy

The model layer is intentionally pluggable. Planned profiles include:

- lightweight multimodal model for low-memory machines;
- standard local VLM for general reasoning + screenshots;
- computer-use-specialized VLM for stronger GUI grounding;
- optional OpenAI-compatible local server adapters (for llama.cpp, vLLM, MLX-backed servers, etc.).

Conway should select the best compatible profile automatically, while allowing advanced users to override it in configuration.

## Safety boundary

Conway is intended to operate with the permissions already granted to the current user account. It deliberately does **not** implement privilege escalation, permission bypass, credential harvesting, stealth/persistence mechanisms, or security-control evasion.

## License

MIT
