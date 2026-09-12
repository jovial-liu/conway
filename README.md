# Conway

Conway is a **local-first, headless autonomous computer-use agent harness**. It has no chat box in its control loop: after you start it, Conway repeatedly observes the desktop, asks a pluggable vision-language model for one next action, executes that action through a cross-platform GUI adapter, records the result, and observes again.

> Status: early v0.3 MVP. Public APIs and model manifests may still change.

## What v0.3 contains

- Screenshot-first VLM observations.
- HiDPI/Retina-aware coordinate mapping between screenshot pixels and OS mouse coordinates.
- Best-effort active application/window metadata for macOS, Windows, and Linux.
- Mouse, keyboard, scrolling, dragging, and Unicode text paste support.
- A continuous observe → decide → act → verify loop.
- File-only state, rolling memory, and JSONL trajectory logs; no SQL or vector database.
- VLM-driven rolling memory compaction with a bounded file fallback.
- Model-output validation and bounded GUI actions before execution.
- Automatic local model profile selection, including a 2B low-memory fallback.
- Local llama.cpp server integration plus any external OpenAI-compatible multimodal endpoint.
- A mock brain for installation/loop testing without downloading a model.
- `conway status` for inspecting a running or previous session without a chat UI.

## Architecture

```text
constitution.md
      │
      ▼
Autonomous Loop ─── Rolling File Context / JSONL Journal
      │
  ┌───┴────────────────┐
  ▼                    ▼
VLM Brain           Computer Adapter
  │                    │
  ▼                    ▼
llama.cpp /         Screenshot / active window /
external API        Mouse / Keyboard / Scroll / Drag
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

Conway selects a bundled local VLM profile from detected RAM/VRAM/unified memory. The local backend expects a current `llama-server`/llama.cpp installation; model files are pulled from Hugging Face by llama.cpp on first launch.

Bundled profiles currently reference:

| Profile | Model | Approx. minimum effective memory |
| --- | --- | ---: |
| `tiny` | `mradermacher/Qwen3-VL-2B-Instruct-abliterated-GGUF` | 5 GB |
| `small` | `mradermacher/Qwen3-VL-4B-Instruct-abliterated-GGUF` | 8 GB |
| `standard` | `mradermacher/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF` | 14 GB |
| `computer-use` | `xlangai/OpenCUA-7B` | externally served |

The abliterated checkpoints are community-modified open model artifacts, not official Qwen safety-tuned releases. OpenCUA is a separate computer-use-specialized model. None of these weights are part of Conway itself.

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

Inspect local state at any time:

```bash
conway status
```

`Ctrl+C` stops the loop. PyAutoGUI's corner-of-screen failsafe remains enabled.

## HiDPI / scaled displays

The VLM always returns coordinates in **screenshot pixels**. Conway stores the screenshot dimensions separately from the operating system's mouse/input coordinate space and maps between them before clicking or dragging. This avoids a common failure mode on Retina displays and Windows systems using display scaling.

## Memory

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

`memory.md` is a rolling Markdown state. The VLM writes concise durable notes during normal cycles. When it becomes large, Conway asks the same model to compact it into a shorter state containing objectives, durable facts, unfinished work, and recurring failures. JSONL trajectories remain available separately for later continual-learning work.

## OS notes

- **macOS:** grant normal Accessibility and Screen Recording permissions when macOS requests them. Conway also makes a best-effort query to System Events for the foreground app/window.
- **Windows:** Conway runs with the permissions of the user who started it and obtains foreground-window metadata through Win32 APIs.
- **Linux:** X11/XWayland is currently the easiest path. If `xdotool` is installed Conway also records active-window metadata. Wayland behavior depends on compositor/session permissions.

Screenshot perception remains the universal fallback when structured desktop metadata is unavailable.

## Security boundary

Conway does not add a confirmation dialog before every GUI action once `--execute` is explicitly used. It also does not implement privilege escalation, UAC/TCC/sudo bypass, credential harvesting, stealth persistence, or OS security-control evasion. The operating system's existing permission model remains the outer boundary.

## Development

```bash
pip install -e ".[dev]"
pytest
```

CI runs on Ubuntu, Windows, and macOS. GitHub is the primary source repository; `jnjnkj/conway` on Hugging Face is automatically mirrored from `main`.

## Next engineering targets

1. Native accessibility/UI-tree adapters beyond active-window metadata.
2. Model-specific GUI action adapters, especially OpenCUA-style grounding outputs.
3. Easier llama.cpp bootstrap for users who do not know their hardware stack.
4. Better multi-monitor handling and platform-native screenshot backends.
5. Optional continual-learning pipelines built from successful JSONL trajectories.

## License

MIT
