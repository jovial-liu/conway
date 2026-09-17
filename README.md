# Conway

Conway is a **local-first, headless autonomous computer-use agent harness**. It does not need a chat box in its control loop: after startup it repeatedly observes the desktop, asks a pluggable vision-language model for one next action, executes that action, records the result, and observes again.

> Status: v0.4 — functional autonomous-agent MVP with cross-platform CI. APIs may still evolve.

## What Conway does

- Continuous **observe → decide → act → verify** loop with no per-step human prompt.
- Screenshot-first VLM perception plus best-effort foreground app/window metadata.
- HiDPI/Retina-aware mapping from screenshot pixels to real OS mouse coordinates.
- GUI actions: click, double-click, move, drag, type/paste, key press, hotkeys, scroll, wait.
- Native current-user tools: shell command, text file read/write, directory listing, URL open.
- Tool-first behavior: the generic VLM is instructed to prefer direct system/file tools when more reliable and fall back to GUI when the interface is the only practical route.
- File-only state and memory: Markdown + JSON + JSONL; no SQL or vector database.
- Rolling VLM memory compaction with a bounded fallback.
- Automatic 2B/4B/8B local VLM selection from detected RAM/VRAM/unified memory.
- llama.cpp local runtime support and arbitrary OpenAI-compatible multimodal endpoints.
- Native **OpenCUA-7B action adapter** with safe AST parsing and Qwen2.5-VL smart-resize coordinate conversion.
- Single-instance lock per state directory.
- `CONWAY_HOME` override for portable/test installations.
- macOS, Windows, and Linux CI.

## Architecture

```text
constitution.md + config.yaml
          │
          ▼
   Autonomous Loop ───── Rolling memory.md / JSONL journal
          │
   ┌──────┴─────────┐
   ▼                ▼
VLM Brain        Tool Executor
   │                │
   │        ┌───────┴────────┐
   ▼        ▼                ▼
Generic / GUI Computer   System Tools
OpenCUA   screenshot      shell / files /
          input           open URL
```

Conway is the harness. Models are replaceable.

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

Initialize local state/config:

```bash
conway init
```

## Hardware and models

```bash
conway doctor
conway models
```

Bundled model profiles:

| Profile | Model | Brain adapter | Approx. minimum effective memory |
| --- | --- | --- | ---: |
| `tiny` | `mradermacher/Qwen3-VL-2B-Instruct-abliterated-GGUF` | generic JSON | 5 GB |
| `small` | `mradermacher/Qwen3-VL-4B-Instruct-abliterated-GGUF` | generic JSON | 8 GB |
| `standard` | `mradermacher/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF` | generic JSON | 14 GB |
| `computer-use` | `xlangai/OpenCUA-7B` | OpenCUA native | external server |

The abliterated Qwen checkpoints are community-modified open model artifacts, not official Qwen safety-tuned releases. OpenCUA is a separate MIT-licensed computer-use model. None of these weights are part of Conway itself.

For automatic local startup, install a current llama.cpp build so `llama-server` or `llama` is on `PATH`. Conway uses llama.cpp's `-hf` model pull path on first launch.

## Run

Smoke-test the loop without downloading a real model:

```bash
conway start --mock --max-steps 3
```

Start with automatic local model selection in observation-only mode:

```bash
conway start
```

Allow Conway to execute enabled actions for the running process:

```bash
conway start --execute
```

Run GUI-only if you do not want shell/filesystem/URL tools in that session:

```bash
conway start --execute --gui-only
```

Connect an existing OpenAI-compatible multimodal model server:

```bash
conway start \
  --endpoint http://127.0.0.1:8000/v1 \
  --model your-model-name \
  --brain generic \
  --execute
```

Use OpenCUA through a compatible endpoint:

```bash
conway start \
  --profile computer-use \
  --endpoint http://127.0.0.1:8000/v1 \
  --model xlangai/OpenCUA-7B \
  --brain opencua \
  --execute
```

`Ctrl+C` stops the process. PyAutoGUI's corner-of-screen failsafe remains enabled.

## OpenCUA grounding

OpenCUA-7B emits pyautogui-style calls such as:

```text
pyautogui.click(x=960, y=324)
```

Those coordinates are in the model's Qwen2.5-VL **smart-resized image**, not directly in the original screenshot. Conway v0.4 reproduces the smart-resize geometry, maps the model point back to the screenshot, then applies its normal screenshot-to-OS HiDPI conversion.

The generated string is **not executed as Python**. Conway parses exactly one allowlisted literal call with Python's AST and translates it into the normal validated action schema.

## Config

The first `conway init` creates `config.yaml` in the local state directory. Inspect it with:

```bash
conway config
```

Default shape:

```yaml
profile: auto
brain: auto
endpoint: null
model: null
port: 8042
interval: 0.5
screenshot_keep: 30
memory_compaction_bytes: 64000
tools:
  gui: true
  shell: true
  filesystem: true
  open_url: true
  max_shell_seconds: 45.0
  max_output_chars: 12000
```

CLI flags override the relevant runtime choices. To relocate all local state:

```bash
export CONWAY_HOME=/path/to/conway-state
```

## Native tools

The generic brain can select one action per cycle from the enabled tool set.

GUI:

```text
click / double_click / move / drag / type / press / hotkey / scroll / wait
```

Local current-user tools:

```text
shell / read_file / write_file / list_dir / open_url
```

Shell commands run with the same OS account and environment that launched Conway, with bounded time and captured output. File tools are text-oriented and size-bounded. Conway does not contain a privilege-escalation mechanism.

## Local state and memory

```text
conway/
├── config.yaml
├── constitution.md
├── memory.md
├── state.json
├── instance.lock
├── llama-runtime.log
├── journal/
│   └── YYYY-MM-DD.jsonl
└── screenshots/
```

Inspect state without a chat UI:

```bash
conway status
```

`memory.md` is a rolling durable state. The VLM can add concise durable notes during normal cycles. Once the file exceeds the configured threshold, Conway asks the generic VLM to compact it into objectives, durable facts, unfinished work, and recurring failures. Raw trajectories remain in JSONL for future continual-learning work.

## OS behavior

- **macOS:** grant normal Accessibility and Screen Recording permissions when requested. Foreground app/window metadata is queried through System Events when available.
- **Windows:** foreground-window metadata uses Win32 APIs. Conway otherwise runs with the privileges of the account that launched it.
- **Linux:** X11/XWayland currently gives the broadest compatibility. If `xdotool` is installed Conway records foreground-window metadata. Wayland input/screenshot behavior depends on compositor policy.

Screenshot perception is the universal fallback when structured desktop metadata is unavailable.

## Security boundary

`--execute` is a one-time process-level opt-in; Conway does not add a confirmation dialog before each action. It also does **not** implement privilege escalation, UAC/TCC/sudo bypass, credential harvesting, stealth persistence, or security-control evasion. The operating system's permission model is the outer boundary.

## Development

```bash
pip install -e ".[dev]"
pytest
```

CI compiles, tests, and runs `conway doctor` on Ubuntu, Windows, and macOS. GitHub is the primary repository; `jnjnkj/conway` on Hugging Face mirrors `main` automatically.

## Next engineering targets

1. Native accessibility/UI-tree snapshots in addition to screenshot + active-window metadata.
2. Better multi-monitor and Wayland-native backends.
3. Automated llama.cpp bootstrap/install assistance.
4. Model-specific adapters beyond generic JSON and OpenCUA.
5. Trajectory scoring/export for future continual learning.
6. Signed desktop packaging and release artifacts.

## License

MIT
