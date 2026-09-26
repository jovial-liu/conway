# Configuration and operational semantics

`conway init` creates state and a default `config.yaml`. Existing files are not reset. `conway config --check` validates the file. Runtime flags override matching config values. Unknown keys, duplicate YAML keys, quoted boolean strings, NaN/infinite values and invalid ranges produce errors.

Important defaults:

| Setting | Default | Meaning |
|---|---|---|
| `profile`, `brain` | `auto`, `auto` | Estimated memory profile and response adapter |
| `endpoint`, `model` | `null` | Local runtime, or an explicitly configured external server |
| `runtime_path` | `null` | Find `llama-server` on PATH; otherwise use supplied absolute path |
| `api_key_env` | `null` | Name of an environment variable, never the API key itself |
| `port`, `context_size` | 8042, 8192 | Local server loopback port and context allocation |
| `interval` | 0.5 | Delay between cycles in seconds |
| `request_timeout`, `startup_timeout` | 180, 900 | Model request / initial runtime wait bounds |
| `max_errors` | 5 | Session failure limit; in continuous mode, threshold for longer recovery backoff |
| `idle_initial_seconds`, `idle_max_seconds` | 2, 60 | Continuous idle backoff range |
| `recovery_initial_seconds`, `recovery_max_seconds` | 5, 300 | Continuous recovery backoff range after the error threshold |
| `repeat_action_limit` | 3 | Same side-effect action allowed on an unchanged observation before suppression |
| `context_chars` | 16000 | Character budget, not a tokenizer-exact context limit |
| `memory_compaction_bytes` | 64000 | Upper ceiling; effective trigger also respects half the context-character budget |
| `screenshot_keep` | 30 | Recent images retained |
| `ui_tree`, `ui_timeout`, `ui_max_nodes` | false, 2, 80 | Optional native UI snapshot controls |
| `opencua_min_pixels`, `opencua_max_pixels` | 3136, 12845056 | Must match the serving model processor |
| `tools.gui/shell/filesystem/open_url` | true | Tools available when execution is enabled |
| `tools.workspace` | null | Default cwd for relative paths; not an access-control boundary |
| `tools.max_shell_seconds` | 45 | Shell command time budget, also bounded by action schema |
| `tools.max_output_chars` | 12000 | File/shell output bound |

`conway run` executes enabled tools continuously by default; use `run --observe` for planning only. The compatibility `start` command requires `--execute` and stops on `finish`. Execution mode is selected at process start, not persisted as a permission in config. No step-by-step conversational approval is inserted. `run` has no chat/task-input option; its ongoing goal comes from the constitution.

Pacing values must be finite numbers between 0.1 and 3600 seconds, with each initial value no greater than its maximum. `repeat_action_limit` must be an integer from 1 to 50. Old config files inherit these defaults without being rewritten.

Example external setup:

```yaml
endpoint: http://127.0.0.1:8000/v1
model: null
brain: auto
api_key_env: CONWAY_MODEL_KEY
interval: 1.0
tools:
  shell: false
  workspace: '/absolute/path/to/workspace'
```

Only configure `api_key_env` when the variable is set. Credentials embedded in endpoint URLs are rejected. If using a remote endpoint, screenshots and supplied context are sent to that service; local-first is not a promise that an explicitly configured remote server receives no data.

## Session controls

`pause`, `resume` and `stop` write a command tagged with the current session ID. An old command does not stop an unrelated new session. Controls are checked before and after inference, between cycles and during cooperative waits or shell polling. Native GUI calls and HTTP requests are not interrupted at arbitrary machine instructions.

`--max-steps` counts attempts, including failed decisions. `--max-seconds` is checked between operations; HTTP timeouts and action bounds still govern an in-flight call. These are cooperative budgets, not a hard CPU or wall-clock sandbox.

## Recovery

State is atomically replaced and the previous valid version is kept. Malformed state is surfaced as `recovery_required`; damaged contents are preserved when the state is rewritten. Before a side effect, the loop saves an action intent with an operation ID. It records the result and clears the pending field only after dispatch returns and the journal is updated. Dispatch failures with a pending intent are terminal rather than blindly retried.

A new run logs the uncertain previous intent and gets a new observation. It does not replay the old action. The model may choose subsequent work after reading the uncertainty; this is not an exactly-once guarantee for arbitrary GUI/system operations. Review the desktop after a crash before restarting execution.

For `start`, `completed` means the model returned `finish` with a completed outcome. It does **not** independently verify the external task. OpenCUA `FAIL` becomes a failed outcome, not completion. For `run`, either outcome increments `goal_reports`, stores `last_goal_report` with `verified: false`, and continues. `idle` and `recovering` are live states with `next_wake_at`, not terminal results. Exported trajectories leave task-success labels unknown.

## Storage and privacy

Memory summaries and tool logs may contain sensitive local information. Screenshots are bounded in count; JSONL journals are split at approximately 5 MB and retained until the owner archives/deletes them. No automatic upload or online training occurs. Export only copies result records to a new local file, excluding dry runs by default and excluding image bytes.

The subprocess environment omits common publishing tokens and the configured model API-key variable. This reduces accidental inheritance; it is not isolation from the user's files, clipboard, shell or GUI. The current-user process has the permissions of its account. Use a disposable account/VM for genuinely untrusted autonomous actions.
