# Device acceptance / 设备验收

This procedure distinguishes installed software, a responding model, synthetic visual localization and a verified desktop task. Completing one stage does not establish the later stages. Run commands in the same activated Python environment. Existing configurations and the constitution remain valid in 0.6.0.

## 1. Offline installation

```sh
python -m pip install -e .
conway init
conway config --check
conway doctor
conway start --mock --max-steps 3
```

No desktop or model is used by the mock run. Edit the constitution for your ongoing scope. Keep `--home PATH` before any subcommand if using a separate state directory.

## 2. Deployment preflight

Configure `endpoint`, `model` and optionally `api_key_env` in config.yaml for an existing multimodal server, or install llama-server and configure `runtime_path` for local inference.

```sh
conway preflight --desktop --output ./preflight.json
```

Preflight does not start a model or download weights. For an external service it queries `/models` and validates the served ID. Local checks cover the runtime executable and estimated memory only. The API key and endpoint URL are omitted from the report. It also checks the constitution, uncertain recovery state and presence of the GUI dependency.

The optional desktop worker captures once, reads display geometry and optional UI-provider status, and removes its temporary screenshot. It never sends the image to the model or dispatches input. It times out after `15 + ui_timeout` seconds. Window titles and UI labels are excluded from the report. Native Wayland is reported as unsupported. Without `--desktop`, desktop status is `not_checked` and `ready_for_observation` is false.

Exit `0` means all checks requested at this stage passed; `2` means at least one check failed; `1` means configuration/report/lock/setup failure. Recovery uncertainty is reported as a warning: inspect the desktop before restarting execution. A passed capture does not verify keyboard/mouse permissions or application behavior.

## 3. Protocol and synthetic vision

```sh
conway probe
conway vision-check --samples 8 --seed 42 --output ./vision.json
```

Both commands may load/download the configured local model, or send synthetic images to the configured external service. Neither reads your desktop. Vision checks use between 1 and 100 cases, with a seed from 0 through 4294967295. The model sees four colored rectangles and a requested color, but no target bounds or answer coordinates. Each case is a fresh inference without previous answers. Supported adapters are generic JSON and OpenCUA, using the configured smart-resize limits.

A hit requires one left-click action within the target rectangle, including its border. The report also gives center error in pixels for valid left clicks. Waits, other actions, right clicks, malformed responses and failed requests count as misses. A hit anywhere inside the target is not proof of pixel-perfect centering. Latency includes request retries. No executor is created and no proposed action is dispatched.

Exit `0` requires all cases to hit. A miss or per-case request error gives exit `2`, while startup failure gives exit `1`. This strict smoke-test threshold is not a statistical claim about general task performance. Eight simple images cannot establish GUI competence. The report has `real_desktop_tested: false` and `actions_executed: 0` even when every case hits.

Reports include suite/Conway version, seed, per-case PNG SHA256, dimensions, target bounds, returned coordinates, hit rate, latency, model ID and processor limits. They do not invent weight revision, quantization or inference-engine version; those fields remain null. Record those deployment details separately for reproducibility. With the same suite version, seed and Pillow environment, cases can be regenerated using `conway.evaluation.make_cases`.

Reports must be new paths outside the state directory. Existing files and symlinks are rejected. Configuration and startup failures do not create a successful report. API error bodies and arbitrary response text are not copied into per-case errors.

## 4. One bounded task

On a test desktop/account, use a disposable work directory and task. First inspect plans:

```sh
conway start --task "Create conway-test.txt in the workspace, write a short note, then read it back to verify" --max-steps 10
```

Then execute the same goal with a finite budget:

```sh
conway start --task "Create conway-test.txt in the workspace, write a short note, then read it back to verify" --execute --max-steps 30 --max-seconds 300
```

For GUI acceptance, repeat a similarly narrow task in a disposable text editor with `--gui-only` and inspect the resulting saved file yourself. Test Chinese input, actual monitor scaling, pause/resume/stop and the corner failsafe. Record your own observed outcome; a model's `finish` action is a model declaration, not an independent success label.

Use `--task-file ./task.md` for a UTF-8 goal instead of `--task`. The goal is bounded to 16000 characters, stored in state/journal and supplied each cycle, subject to the constitution. It does not edit constitution.md. Every new start sets its own task, with no automatic reuse of the prior assignment. Existing memory and journal context remain available as historical observations.

## Evidence to retain

Keep preflight and vision JSON, OS/display details, exact weights/quantization/engine version, the session task and journal, and a manually verified result for the disposable file. Keep raw screenshots local if needed for your own review; the diagnostic reports do not contain them. Real Mac/Windows/X11 input and real-model accuracy still require this device procedure; CI fixtures do not substitute for it.
