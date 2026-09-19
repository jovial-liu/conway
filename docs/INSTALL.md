# Installation and model startup

## 1. Install Conway

Python 3.11+ is required. Clone the source, create a virtual environment, and run `python -m pip install -e .`. The `conway` console entry point and `python -m conway` are equivalent. A built `conway_agent-0.5.0-py3-none-any.whl` installs with `python -m pip install PATH_TO_WHEEL`; runtime Python dependencies still need to be installed.

On Windows, use `.venv\Scripts\python.exe` and `.venv\Scripts\conway.exe` directly instead of changing execution policy. On macOS/Linux activate the environment with `source .venv/bin/activate`.

Run `conway init`, `conway doctor`, then `conway start --mock --max-steps 3`. The latter works without a display or GPU. `conway init` prints the editable config and constitution locations.

## 2. Install or connect the inference engine

Conway does not install GPU drivers, Homebrew, WinGet or llama.cpp itself. It never runs an unsolicited installer.

- **Apple Silicon:** an existing Homebrew installation can install llama.cpp with `brew install llama.cpp`. Confirm `llama-server --help` works in the same terminal/environment as Conway. A Metal-capable build is needed for GPU acceleration.
- **Windows:** obtain a suitable CPU/CUDA llama.cpp release from the project's official Releases page. Extract the archive, keep its DLLs beside `llama-server.exe`, and set an absolute `runtime_path` in config when the executable is not on PATH. Choose a build compatible with the GPU driver; hardware detection does not validate the driver/build combination.
- **Linux:** use an official compatible binary or build llama.cpp for the target CPU/CUDA backend. Place `llama-server` on PATH or set `runtime_path`. Ordinary package dependencies and desktop capture utilities vary by distribution.
- **AMD/Intel or other inference stacks:** use a compatible already-running multimodal endpoint. Automatic driver installation and automatic ROCm/SYCL/Vulkan selection are not implemented.

Sources: [llama.cpp](https://github.com/ggml-org/llama.cpp), [server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [multimodal support](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).

## 3. Local model profiles

`conway models` shows the bundled choices. Auto mode selects an estimated fitting profile using available memory, not total installed RAM alone. First launch lets llama.cpp download the referenced Q4_K_M model and compatible visual projection assets through `-hf`. You need network access and sufficient disk space for that download. Not every third-party model/backend combination has been verified with actual inference for this release.

Run `conway probe` before a desktop session. It requires a valid model list, sends a synthetic PNG and parses the returned action without executing it. A `protocol_ok` result means the request and output schema worked, not that visual grounding is accurate or even that the model used the pixels meaningfully.

To use a specific local profile: `conway start --profile tiny`. Insufficient estimated memory produces an error rather than attempting a model that cannot fit. Close other applications or configure an external endpoint. An 8 GB machine is not automatically guaranteed to fit the tiny VLM: the desktop and other applications consume memory too.

Conway starts its own server on loopback only, alias `conway-local`, default port 8042. An occupied port is an error. Reusing an existing server requires explicitly supplying `--endpoint`; it will not silently attach to the wrong model. Runtime logs are in `llama-runtime.log` under the state directory. `startup_timeout` can be raised for a slow first download.

## 4. OpenCUA

Serve `xlangai/OpenCUA-7B` using a backend supported by its [official model card](https://huggingface.co/xlangai/OpenCUA-7B). The official vLLM instructions require the model's custom-code support; review that repository before enabling it. The serving command belongs to the inference server, not Conway. Its served model name must appear in `/v1/models`.

Then connect:

```sh
conway probe --endpoint http://127.0.0.1:8000/v1 --brain opencua
conway start --endpoint http://127.0.0.1:8000/v1 --brain opencua --execute
```

If needed, add `--model opencua-7b` using the actual served alias. Set `opencua_min_pixels` and `opencua_max_pixels` to match the server's image processor. Defaults follow the model card (3136 and 12845056, factor 28). This adapter uses one screenshot per decision and a strict single-literal-call subset; it is not the official multi-screenshot benchmark harness. Unsupported multiline programs are rejected, never executed.

## 5. Desktop prerequisites

- macOS needs the normal Accessibility and Screen Recording grants for the terminal/interpreter that actually launches Conway. Window metadata through System Events can also require Automation permission. Restart the launching application after granting permissions when macOS requires it.
- Windows requires an interactive desktop session. Elevated applications and the secure desktop remain subject to the OS boundary; Conway does not bypass UAC.
- Linux's current GUI backend targets X11. XWayland does not imply full control over native Wayland applications. Headless SSH sessions cannot control a nonexistent desktop; `--mock` remains available.

The primary display is the supported coordinate target. Do not infer multi-monitor support from successful unit tests of scaling arithmetic. Unicode paste needs a working clipboard implementation; unavailable clipboard support is reported instead of pretending the text was typed.

## 6. Optional UI snapshots

Install `python -m pip install -e '.[accessibility]'` for the platform-specific Windows/macOS dependencies, then set `ui_tree: true`. On Linux, compatible system `pyatspi`/AT-SPI bindings must be visible to the selected Python environment; this is distribution-specific and is not installed automatically. Keep `ui_tree: false` when those bindings are unavailable.

Snapshots are read-only, capped by `ui_max_nodes`, and time limited by `ui_timeout`. Missing permissions or dependencies produce `unavailable`; blocked providers produce `timeout`. Screenshot operation remains the fallback. These adapters have not been exercised against real native desktops in this release's local validation.
