# Conway

Conway is a **local-first, headless autonomous computer-use agent harness**. It has no chat box in its control loop: after you start it, Conway repeatedly observes the desktop, asks a pluggable vision-language model for one next action, executes that action through a cross-platform GUI adapter, records the result, and observes again.

> Status: early v0.2 MVP. The public APIs and model manifest may change.

## What v0.2 contains

- Screenshot-first VLM observations.
- Mouse, keyboard, scrolling, dragging, and Unicode text paste support.
- A continuous observe → decide → act → verify loop.
- File-only state, memory, and JSONL trajectory logs; no SQL or vector database.
- Model-output validation and bounded GUI actions before execution.
- Automatic local model profile selection based on available memory.
- Local llama.cpp server integration plus any external OpenAI-compatible multimodal endpoint.
- PyAutoGUI desktop backend for macOS, Windows, and Linux.
- A mock brain for installation/loop testing without downloading a model.

## Architecture

```text
constitution.md
      │
      ▼
Autonomous Loop ───── File Context / Recent Journal
      │
  ┌───┴─────────────┐
  ▼                 ▼
VLM Brain        Computer Adapter
  │                 │
  ▼                 ▼
llama.cpp /      Screenshot / Mouse /
external API     Keyboard / Scroll / Drag
```

Conway is the harness. The VLM is replaceable.

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/jovial-liu/conway.git
cd conway
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -e .
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -e .
```

## Hardware detection

```bash
conway doctor
```

Conway currently selects between bundled local VLM profiles using detected RAM/VRAM/unified memory. The local backend expects a current `llama-server`/llama.cpp installation; model files are then pulled by llama.cpp from Hugging Face on first launch.

Bundled profiles currently reference:

- `mradermacher/Qwen3-VL-4B-Instruct-abliterated-GGUF` for lower-memory machines.
- `mradermacher/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF` for stronger machines.
- `xlangai/OpenCUA-7B` as an optional computer-use-specialized model served through a compatible external endpoint.

These model repositories are third-party/open model artifacts and are not part of Conway itself.

## Run

Test the loop without a real model:

```bash
conway start --mock --max-steps 3
```

Start with automatic local model selection in observation-only mode:

```bash
conway start
```

Allow Conway to execute its selected GUI actions after the one-time process start:

```bash
conway start --execute
```

Use an already-running OpenAI-compatible multimodal server:

```bash
conway start \
  --endpoint http://127.0.0.1:8000/v1 \
  --model your-model-name \
  --execute
```

`Ctrl+C` stops the loop. PyAutoGUI's corner-of-screen failsafe remains enabled.

## Local state

The default per-user state directory contains:

```text
conway/
├── constitution.md
├── memory.md
├── state.json
├── llama-runtime.log
├── journal/
│   └── YYYY-MM-DD.jsonl
└── screenshots/
```

The journal is intentionally plain JSONL so later continual-learning work can transform successful trajectories into datasets without migrating a database first.

## OS notes

- **macOS:** grant the normal Accessibility and Screen Recording permissions when the OS requests them.
- **Windows:** Conway runs with the permissions of the user who started it; it does not require administrator privileges for ordinary desktop control.
- **Linux:** X11/XWayland is the easiest current path. Wayland behavior depends on compositor/session permissions and will get a native adapter later.

## Security boundary

Conway does not add a confirmation dialog before every GUI action once `--execute` is explicitly used. It also does not implement privilege escalation, UAC/TCC/sudo bypass, credential harvesting, stealth persistence, or OS security-control evasion. The operating system's existing permission model remains the outer boundary.

## Development

```bash
pip install -e ".[dev]"
pytest
```

GitHub is the primary source repository. `jnjnkj/conway` on Hugging Face is maintained as a mirror/distribution page.

## Next engineering targets

1. Native accessibility/UI-tree adapters for macOS, Windows, and Linux.
2. Better GUI grounding profiles and model-specific action adapters.
3. Easier llama.cpp runtime bootstrap for users who do not know their hardware stack.
4. Context compaction driven by the VLM rather than simple bounded file history.
5. Optional continual-learning pipelines built from the JSONL trajectories.

## License

MIT
