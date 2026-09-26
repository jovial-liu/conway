# Keep the harness small; let the model improve

Conway's long-term goal is digital-life research. Better small models, continual
learning and recursive improvement are possibilities to investigate, not a
capability or a delivery date that the runtime can promise today.

## Default operating model

The running system has one acting policy and one loop:

**Read goals → observe → decide → validate/act → record → observe again.**

`constitution.md` holds identity, scope and operating rules. `goals.md` holds the
owner's current work and long-term direction, reread on every continuous cycle.
`memory.md`, `state.json` and the JSONL journal preserve context and progress.
No database, chat window, agent hierarchy or training scheduler is required.
Existing constitution/goals files are preserved by initialization and upgrades.

Use [the digital-life goals example](../examples/goals.digital-life.md) as the
contents of the `goals.md` path printed by `conway init`. Editing it does not
interrupt an action already in flight; it affects the next decision cycle.
Keep it under 8,000 characters. Removing/emptying it leaves the constitution in
charge. Single-session `start --task` deliberately ignores ongoing goals so a
bounded acceptance test does not inherit unrelated research work.

MCP, Agent Skills, episode recording and training scripts remain optional. New
model capabilities should normally replace the policy adapter, not introduce
another orchestration framework around the main loop. Runtime command setup is
headless, including disabling the local llama-server Web UI.

## The minimal model boundary

Current adapters implement `Brain.decide(constitution, memory, observation)` and
return one `Decision`. Native model output formats are translated to the shared
action schema. `close()` releases resources. Memory compaction is optional.

Version 0.9 adds one optional method:

```python
def feedback(self, transition: Transition) -> None:
    ...
```

`Transition` provides session/operation IDs, cycle, prior observation, executed
action, tool result, next observation (when available), a terminal flag and
`task_success=None`. It supplies experience, not a reward invented by the acting
model. The current HTTP/generic/OpenCUA adapters inherit a **no-op**: this release
does not send a new training request or update weights in the background.

A future stateful or learning adapter may consume that transition according to
its actual model API. Feedback is delivered before the next decision, after a
fresh observation. Normal session/step completion delivers the remaining item
with no next observation. Dry runs, uncertain partial actions and suppressed
actions produce no learning transition. Explicit stop does not initiate terminal
learning work. A feedback error stops that adapter without replaying the action.

The hook is synchronous, process-local and at most once per operation in a run.
It is not crash-durable delivery. Image paths are temporary and subject to
retention; consume/copy them before returning. An adapter that persists or learns
must implement its own checkpoint atomicity and deduplication using operation
IDs. Existing opt-in episodes provide the separate, durable offline data route.

## Prepare the model and the runtime independently

| Component | Today's artifact | Future replacement point |
|---|---|---|
| Decision model | Existing VLM weights and inference adapter | A more capable or stateful model behind `decide` |
| Experience | Executed transitions and opt-in reviewed visual episodes | Actual continual-learning API behind `feedback` |
| Long-term memory | Human-readable files | A model with native memory can consume the same goals and evidence |
| Harness | Versioned Python code and tool/action contracts | A candidate code revision evaluated with a fixed model |
| Model release | Base revision, weight hashes, data/training manifests | A candidate checkpoint evaluated with a fixed harness |

There is no generic `enable_recursive_evolution=true` setting: mechanisms and
interfaces depend on future models. Avoid speculative learner registries,
population managers and auto-training daemons before a real model requires them.

## Versioned improvement experiments

Keep the live baseline and the experimental candidate separate. An experiment
record should identify parent model/harness revision, candidate revision, task
set, input data, compute/time budget, results and rollback target. Reuse the same
held-out tasks for a model comparison, and the same model for a harness comparison.

A candidate that changes its own score labels, drops failed tasks or repeats a
self-report has not demonstrated improvement. Independent artifact checks,
unseen tasks and retention of old capabilities are necessary evidence. Replacement
should happen at a clean session boundary using the experiment's configured
release policy; automatic hot replacement is not implemented here.

The bounded copy-and-launch acceptance case checks only a local program-copy
component. It neither installs Conway into other machines nor recursively starts
agents. Successful copying is distinct from continual learning and recursive
capability improvement. See [model development](MODELS.md) for the existing
offline path and [validation scope](VALIDATION.md) for what was actually tested.
