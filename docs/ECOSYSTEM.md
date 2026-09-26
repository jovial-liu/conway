# Ecosystem integration

Version 0.8 supports local Agent Skills and optional MCP **stdio tools** for the generic JSON policy. These capabilities run inside the existing autonomous loop. There is no chat UI or per-action confirmation prompt.

## Agent Skills

Place reviewed packages at `<CONWAY_HOME>/skills/<name>/SKILL.md`. Additional search directories can be set in `config.yaml`:

```yaml
skill_dirs:
  - /absolute/path/to/my-skills
```

Each SKILL.md uses the [Agent Skills format](https://agentskills.io/specification): YAML frontmatter with a directory-matching lowercase `name`, a non-empty `description`, followed by Markdown instructions. Example contents for `verify-output/SKILL.md`:

```markdown
---
name: verify-output
description: Inspect the resulting file or application state before reporting task success.
---
Read the expected artifact or obtain a fresh observation. Compare it with the goal.
If the evidence is incomplete, record that uncertainty and choose the next check.
```

Run `conway skills` to list names/descriptions without running code. During execution, the generic model sees the catalog and can choose:

```json
{"type":"read_skill","args":{"name":"verify-output","resource":"SKILL.md","offset":0}}
```

The result includes the skill directory and `next_offset` for further pages. Resources such as `references/guide.md` are read by the same action; scripts are not auto-executed. Skill guidance remains subordinate to the constitution. Experimental `allowed-tools` metadata never changes Conway's configured permissions.

There are at most 32 packages, 64 KB per text resource and normally 4,000 characters per page (smaller with a small context budget). The most recent extension result receives a distinct context budget so large journal events do not silently replace the loaded instructions with a tiny excerpt. Reading another extension result replaces that slot; use file memory and reload relevant pages as needed.

Duplicate names, malformed metadata and paths escaping the skill directory are rejected. Metadata/main instructions are snapshotted at startup; restart to discover changes. This is a format-compatible local loader, not a package marketplace or a certification of third-party skills.

## MCP tools

Install the optional client:

```sh
python -m pip install -e '.[mcp]'
```

Configure trusted, already-installed stdio servers. The executable and arguments are separate; Conway does not pass this command through a shell or install it automatically.

```yaml
mcp_servers:
  workspace:
    command: /absolute/path/to/python
    args: [/absolute/path/to/workspace_server.py]
    allowed_tools: [search_files, inspect_artifact]
    env_keys: []
    timeout: 30
```

Replace both paths and tool names with the server's actual configuration. On Windows, use an absolute executable path and YAML single quotes or forward slashes. `env_keys` lists explicitly forwarded environment-variable names; credentials are never placed in this config by Conway. The official SDK otherwise uses a minimal environment allowlist.

`conway mcp-check` **starts the configured programs**, initializes/discovers their tools and prints the exposed catalog without invoking tools. Use it after reviewing server configuration. `conway run` connects the configured servers when real generic-policy execution is enabled. `--mock`, `--observe`, `--gui-only`, and the native OpenCUA adapter do not start MCP servers.

The model can then choose an action such as:

```json
{"type":"mcp_call","args":{"server":"workspace","tool":"inspect_artifact","arguments":{"path":"output.txt"}}}
```

Sessions persist across calls in a run. Conway lists paginated tools, exposes only explicit allowlisted names, validates arguments against the advertised JSON Schema before dispatch, and records intent before calling. Calls count as potential side effects even when the server annotates them as read-only. Tool descriptions/results are untrusted data, not authority to replace the constitution.

Timeouts, disconnects, MCP error results or requests for additional user input stop the run with uncertain intent. They are not automatically retried. Stop interrupts an in-flight call and closes owned connections. A cancellation cannot undo a remote side effect. Normal shutdown closes owned server processes through the official SDK; server-spawned external services are outside that guarantee.

Supported/tested boundary: official Python SDK **2.2**, real local stdio fixture server, discovery, schemas, text/structured output, session continuity, explicit environment forwarding, timeout and shutdown. Binary content is omitted rather than injected into model context; resource links are not fetched. Remote HTTP/OAuth, resources/prompts, dynamic catalog refresh, server sampling, user elicitation and arbitrary third-party servers are not validated in this release. Catalog limits are 8 servers, 64 allowlisted tools per server and 48,000 characters in total.

## Compatibility matrix

| Capability | Generic VLM | Native OpenCUA | Mock |
|---|---|---|---|
| Continuous loop | Yes | Yes | Synthetic smoke only |
| GUI action schema | Yes | Translated literal calls | Synthetic observation |
| File/Shell actions | Configured tools | Not emitted by this adapter | Disabled |
| Agent Skills actions | Yes, except GUI-only | Not emitted by this adapter | Disabled |
| MCP stdio tools | Execution mode, except GUI-only | Disabled | Disabled |
| Visual episode capture | Opt-in | Not implemented | Fixture-only; excluded from training |

The workspace path is a working directory, not a security sandbox. Configured server programs and enabled Shell tools run as the current OS user. Set environment permissions and the constitution accordingly.
