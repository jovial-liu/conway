# Conway Policy: the model development track

**Release status:** no Conway-trained policy weights are available from this change. The Hub repository publishes source, not a newly trained foundation model. Version 0.8 adds a concrete data path and an experimental offline training script alongside the harness.

## Foundation versus policy

An existing VLM supplies perception/language priors; task-specific training should teach reliable action selection, verification, recovery and useful long-horizon behavior. Do not pretrain a foundation model merely to put a project name on it.

The initial experimental recipe defaults to [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), whose official model card describes a small vision-capable model intended for prototyping/task-specific research. It is a starting point, not a measured winner or a promise of desktop competence. Larger official Qwen candidates can be compared when hardware/data permit. [OpenCUA](https://opencua.xlang.ai/) remains a specialist baseline through its native adapter; its action convention differs from Conway's generic JSON output.

The existing hardware-selected GGUF inference profiles have not been replaced with unmeasured training candidates. Inference memory estimates do not establish fine-tuning VRAM requirements. Check the base model and data licenses separately before distributing derivative weights.

## 1. Record a bounded real episode

```sh
conway run --max-steps 100 --record-episode ./episodes/session-001
```

The output directory must be new and outside CONWAY_HOME. Recording is **off by default**. It preserves screenshots, full decision context/constitution, canonical actions, concise action summaries, results and hashes. It does not preserve private chain-of-thought. These records may contain private screen/file content; nothing is uploaded automatically.

The recorded system and user instructions share the generic runtime prompt builder. The assistant target is the normalized action JSON, not arbitrary raw model text. The current recorder covers completed action-result events, not all failed model outputs or uncertain partial actions. Outcome remains `task_success: null` even when the agent emits `finish`.

There is a 1,000-step / 256 MB screenshot limit per recording. Hitting a recording limit or write failure ends that run so recording loss is visible. A crashed `recording` episode cannot be exported until its incomplete state has been investigated. Retain original episodes for research provenance.

## 2. Add external review

Create `review.json` inside the episode. Use its actual ID and step IDs from `episode.json`:

```json
{
  "schema_version": 1,
  "episode_id": "ACTUAL_EPISODE_ID",
  "source": "human",
  "reviewer": "reviewer-or-evaluator-version",
  "privacy_reviewed": true,
  "steps": {
    "ACTUAL_STEP_ID": {
      "accepted": true,
      "outcome": "passed",
      "evidence": "evidence/check-001.txt"
    }
  }
}
```

Evidence must be a non-empty file within that episode. `source` is `human` or `external_evaluator`; no acting-model self-report is an accepted source. Omitted/rejected steps are excluded. Review each accepted step's action quality, not just the terminal outcome. Keep evaluation code/version, criteria and task/environment identity in the evidence.

These annotations are operator assertions with file hashes, **not proof of reviewer independence or automatic factual verification**. The program verifies structure/integrity, not whether an evidence statement is true. Review all prompts, images and file contents for privacy. If sanitization changes an episode, create and review a new derived episode with retained provenance instead of treating old hashes as valid.

## 3. Export and validate training data

```sh
conway dataset --episodes ./episodes/session-001 ./episodes/session-002 --output ./datasets/policy-v1
python scripts/train_policy.py --dataset ./datasets/policy-v1 --validate-only
```

Exports contain Hugging Face/TRL-style multimodal messages, relative PNG paths, review provenance and a hashed dataset manifest. Mock, dry-run, native-OpenCUA, unreviewed, recording-in-progress and corrupt episodes are rejected. Splits are deterministic by episode ID (default 20% validation). Exact duplicate screenshots across splits are rejected. Entire episodes stay together; similar task templates are **not** automatically deduplicated. Collect enough distinct episodes for both splits and hold out tasks/applications manually.

`--validate-only` needs no training libraries/GPU, loads no model and trains nothing. It verifies exported file hashes, counts, message shape, images and split separation, and reports whether both splits are populated.

## 4. Train a candidate explicitly

On a suitable CUDA machine, install the separately optional training dependencies and use an immutable base-model revision:

```sh
python -m pip install -e '.[training]'
python scripts/train_policy.py --dataset ./datasets/policy-v1 --output ./runs/policy-001 --model Qwen/Qwen3.5-0.8B --revision ACTUAL_40_CHARACTER_COMMIT --max-steps 100
```

The recipe uses TRL 1.14.0, Transformers 5.17.0 and PEFT LoRA, with batch size 1, gradient accumulation 8, rank 16 and completion-only loss. It converts the exported messages into prompt/completion form, so context/observations are not training targets. It retains full multimodal sequences (`max_length=None`) to avoid cutting image tokens; long contexts/large images can exceed available VRAM. The exact training environment is recorded in `training.json`.

It saves a candidate adapter and held-out loss metrics, without Hub uploads, runtime weight replacement or automatic promotion. No GPU optimization step has been executed for this release; dependency/API review, data validation and fixture tests do not establish successful fine-tuning, model quality or hardware fit. The recipe may require processor/memory tuning on the chosen GPU and base revision.

## 5. Evaluate before calling it a Conway model release

| Evaluation | Why it matters |
|---|---|
| JSON validity and enabled-tool/schema compliance | A policy must produce executable actions consistently |
| Screenshot grounding, coordinates and DPI | Visual decisions must refer to the actual device environment |
| Externally verified task success | Action execution and model self-reports are insufficient |
| Multi-goal continuity and interruption recovery | Short-task success may not survive long autonomous runs |
| Tool errors, repeated actions and useful work per time/token | Prevent a learned policy from merely producing more activity |
| Fixed-model / fixed-harness ablations | Attribute improvement to model training versus runtime scaffolding |
| Retained performance and unseen environments | Detect overfitting and regression from fine-tuning |

Release a model only with base revision, adapter/weight artifacts, licenses, data lineage, training settings, hardware, independent task results and limitations. A lower validation loss alone is not a digital-life or useful-autonomy result. The [architecture roadmap](ARCHITECTURE.md) describes the longer-term research program.
