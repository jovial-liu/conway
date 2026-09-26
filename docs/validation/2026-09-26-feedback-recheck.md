# 2026-09-26: feedback-context recheck / 反馈上下文复测

**No observed task-success gain.** This is an actual CPU run of the same pinned
Huihui Qwen3.5 0.8B abliterated GGUF used in the
[v0.9 acceptance](2026-09-26-small-models.md), with the v0.10 decision-context changes.
It does not demonstrate learning, self-improvement or digital life.

| Metric | v0.9 | v0.10 |
|---|---|---|
| Synthetic localization, seed 42 | 1/4 | 1/4 |
| File write/readback | Failed; 10 writes, no readback | Failed; 5 writes, no readback |
| Trusted copy/launch | Failed; two response/action errors | Failed; six waits, no copy or launch |
| Workbench cases passed | 0/2 | 0/2 |
| File case elapsed | 130.96 s | 192.28 s |
| Copy case elapsed | 22.52 s | 197.87 s |

中文结论：最新结果优先呈现后，这款 0.8B 模型仍未能利用反馈完成多步任务。
本次输出的动作通过了解析检查，但它重复写入或者持续等待，未获得任务成功。
时间更长也不能解读为模型表现更好。本次修改的确定收益是上下文呈现和重复保护
的工程行为，而不是已经测得的模型能力提升。

## Settings and scope

Same Q4_K_M file and F16 projector, pinned conversion revision, runtime
`llama.cpp b11146 / 7fe450e19`, CPU with four threads, context 8192,
minimum image tokens 1024, reasoning off, one server slot and no Web UI.
Full weight hashes and revision are retained in the report below.
Python 3.12.14 / Pillow 12.3.0; generic JSON policy, temperature 0.1,
max output tokens 1000, request timeout 90 seconds; sampling seed not fixed.

The suite used the same four localization images and two workbench tasks,
10 steps / 180 seconds per case, stopping after two consecutive errors.
Both v0.10 workbench runs instead hit their time budget. An in-flight inference
finished after the budget and its decision was discarded, hence one more cycle
than dispatched actions and elapsed times above 180 seconds.

The report comparison marks budget metadata as unknown: these original JSON
reports did not embed it; the budgets above were checked against the launch
parameters. Current `acceptance-check` outputs now embed budgets, and the
comparison rejects mismatched budgets when both are present.

Workbench uses single-session task semantics, actual bounded file I/O and
synthetic screenshots. Its single-session repeat behavior is intentionally
unchanged; the new non-GUI repeat-count fix applies to `conway run` and is
verified separately by a changing-screen fixture. Neither workbench case
launches a child in this measured run. No real native desktop, 2B recheck,
online training, long-duration autonomy or open-ended replication was tested.

The model, prompts and tasks had already been examined during development.
This is a debugging recheck, not held-out generalization evidence. Two individual
runs on a shared CPU host cannot attribute timing or sampled-output differences
solely to the harness. No accuracy or speed gain is claimed.

## Reproduce and compare

Use the pinned download/server commands in the v0.9 report, then install the
commit containing this v0.10 report. Run the same `vision-check --samples 4
--seed 42` and `acceptance-check --max-steps 10 --max-seconds 180` with new output
paths. The original recheck invoked those Python suite functions directly.

```sh
python scripts/compare_acceptance.py baseline-workbench.json candidate-workbench.json
```

The comparison below is case-level and descriptive. A valid comparison that
finds no gain exits 0; that is not approval to promote a model. Raw inputs and
hashes are retained. See [digital-life research gaps](../DIGITAL_LIFE.md).

## Harness verification

238 local regression tests passed, including large-history context retention,
latest error/dry-run distinction, unchanged full journals, GUI count reset,
non-GUI count retention across changing images, and report comparisons that
reject missing tasks/changed checks/inconsistent labels/mismatched budgets.
Compilation and wheel building passed. CI checks the same release on three OSes.
These are software checks, not evidence of real-model adaptation.

## Comparison output

```json
{
  "schema_version": 1,
  "suite": "conway-workbench-v1",
  "status": "no_observed_gain",
  "cases": 2,
  "baseline_passed": 0,
  "candidate_passed": 0,
  "gained": [],
  "lost": [],
  "candidate_passed_all": false,
  "matching_budget_metadata": null,
  "baseline": {
    "model": "conway-huihui-0.8b",
    "conway_version": "0.9.0",
    "created_at": "2026-09-26T13:55:08.999617+00:00"
  },
  "candidate": {
    "model": "conway-huihui-0.8b",
    "conway_version": "0.10.0",
    "created_at": "2026-09-26T14:20:11.645937+00:00"
  },
  "automatic_promotion": false,
  "note": "Descriptive comparison of supplied reports, not an authenticated evaluator or proof of learning. Control weights, tasks, budgets and runtime; repeat on held-out tasks before drawing conclusions.",
  "input_sha256": {
    "baseline": "4acbeee3b107e1f1c1699c9ad58af7a5870f92e640ee1915b081cc6176dd3c88",
    "candidate": "d644948cbfcd9ed41f5ca7b100239ca66ee1f77bb2a97d38bcafea294f9b1972"
  }
}
```

## Recorded model output

```json
{
  "workbench": {
    "schema_version": 1,
    "suite": "conway-workbench-v1",
    "conway_version": "0.10.0",
    "created_at": "2026-09-26T14:20:11.645937+00:00",
    "model": "conway-huihui-0.8b",
    "brain": "generic",
    "profile": "external",
    "status": "failed",
    "cases": [
      {
        "case": "file_roundtrip",
        "loop_status": "stopped",
        "cycles": 6,
        "error_count": 0,
        "elapsed_seconds": 192.284,
        "actions": [
          "write_file",
          "write_file",
          "write_file",
          "write_file",
          "write_file"
        ],
        "errors": [],
        "model_reported_completion": false,
        "passed": false,
        "checks": {
          "exact_file_content": true,
          "readback_performed": false
        },
        "output_sha256": "a87c1886e3ab7a0e8e75996d9c9b80d904a0dabf412569cdc00a26eb797913a4",
        "synthetic_observation": true,
        "real_model_expected": true,
        "child_launches": 0,
        "max_child_launches": 0
      },
      {
        "case": "bounded_replica",
        "loop_status": "stopped",
        "cycles": 7,
        "error_count": 0,
        "elapsed_seconds": 197.866,
        "actions": [
          "wait",
          "wait",
          "wait",
          "wait",
          "wait",
          "wait"
        ],
        "errors": [],
        "model_reported_completion": false,
        "passed": false,
        "checks": {
          "exact_program_copy": false,
          "seed_read": false,
          "child_exit_and_output": false
        },
        "output_sha256": null,
        "synthetic_observation": true,
        "real_model_expected": true,
        "child_launches": 0,
        "max_child_launches": 1
      }
    ],
    "real_desktop_tested": false,
    "open_ended_self_replication_tested": false,
    "note": "Bounded local component checks with synthetic observations and independently checked files/child output; not proof of autonomous self-replication."
  },
  "vision": {
    "schema_version": 1,
    "suite": "conway-grounding-v2",
    "conway_version": "0.10.0",
    "created_at": "2026-09-26T14:21:37.194871+00:00",
    "seed": 42,
    "model": "conway-huihui-0.8b",
    "brain": "generic",
    "profile": "external",
    "processor": {
      "opencua_min_pixels": 3136,
      "opencua_max_pixels": 12845056
    },
    "model_revision": null,
    "quantization": null,
    "inference_engine_version": null,
    "status": "failed",
    "cases": [
      {
        "case_id": 1,
        "instruction": "Click the center of the blue rectangle. Return one left click. This is a synthetic test image.",
        "image_sha256": "7c69bc9ac5c4156ec83b345fa564b18b1e1a1ae47eddf8215bc57efa598c9737",
        "image_size": [
          840,
          560
        ],
        "target_bounds": [
          487,
          100,
          661,
          155
        ],
        "hit": false,
        "valid_click": true,
        "action_type": "click",
        "point": [
          100,
          100
        ],
        "center_error_pixels": 474.797,
        "latency_seconds": 23.9187
      },
      {
        "case_id": 2,
        "instruction": "Click the center of the red rectangle. Return one left click. This is a synthetic test image.",
        "image_sha256": "56d8c23d4172ef9bf5b7f263a82c07945c11d0b597b4c1ea228c867373e879d9",
        "image_size": [
          1120,
          700
        ],
        "target_bounds": [
          590,
          396,
          827,
          452
        ],
        "hit": false,
        "valid_click": true,
        "action_type": "click",
        "point": [
          112,
          112
        ],
        "center_error_pixels": 673.169,
        "latency_seconds": 23.4838
      },
      {
        "case_id": 3,
        "instruction": "Click the center of the red rectangle. Return one left click. This is a synthetic test image.",
        "image_sha256": "e07d989da8ffb317461990b8a2b4b92e67bb5e7e494423aba50b78e7c4d21b50",
        "image_size": [
          700,
          840
        ],
        "target_bounds": [
          70,
          91,
          225,
          148
        ],
        "hit": true,
        "valid_click": true,
        "action_type": "click",
        "point": [
          100,
          100
        ],
        "center_error_pixels": 51.347,
        "latency_seconds": 19.0045
      },
      {
        "case_id": 4,
        "instruction": "Click the center of the red rectangle. Return one left click. This is a synthetic test image.",
        "image_sha256": "f9f3c34421a0c6764d295291d8a2c8cab8edd147f81448d7ce179999a67559c1",
        "image_size": [
          840,
          560
        ],
        "target_bounds": [
          484,
          88,
          609,
          161
        ],
        "hit": false,
        "valid_click": true,
        "action_type": "click",
        "point": [
          100,
          200
        ],
        "center_error_pixels": 452.838,
        "latency_seconds": 19.0792
      }
    ],
    "samples": 4,
    "hits": 1,
    "hit_rate": 0.25,
    "valid_clicks": 4,
    "errors": 0,
    "median_latency_seconds": 21.2815,
    "actions_executed": 0,
    "real_desktop_tested": false,
    "note": "Synthetic color-target localization only. This does not measure real application task success."
  },
  "provenance": {
    "repo": "mradermacher/Huihui-Qwen3.5-0.8B-abliterated-GGUF",
    "revision": "2fabc82874616f44cdc494ec8ddc0e8ee10654b3",
    "quantization": "Q4_K_M",
    "projector": "F16",
    "runtime": "llama.cpp b11146 / 7fe450e19",
    "threads": 4,
    "context_size": 8192,
    "image_min_tokens": 1024,
    "reasoning": "off",
    "device": "CPU",
    "weights": {
      "Huihui-Qwen3.5-0.8B-abliterated.Q4_K_M.gguf": {
        "bytes": 527503840,
        "sha256": "411e0f945a5d57c63f33bcb5bfa4d5c2711d1ce28cb7635201ac0c311a7dfdf0"
      },
      "Huihui-Qwen3.5-0.8B-abliterated.mmproj-f16.gguf": {
        "bytes": 204987168,
        "sha256": "97f83d38af8508f8ea48b6fe10f120773b5a8760616a5c43f0f2d049ed4b3c90"
      }
    }
  }
}
```
