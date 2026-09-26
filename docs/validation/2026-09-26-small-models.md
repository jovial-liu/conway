# 2026-09-26: real small-model acceptance / 真实小模型验收

**Conclusion: neither tested deployment passes Conway's current acceptance checks.**
Both accepted an image and returned a valid protocol action on the probe. That
alone did not translate into reliable localization or multi-step tool use.
These are actual downloaded weights running on CPU, not scripted model fixtures.

中文结论：两款消融模型均未通过本次任务验收。0.8B 能写对指定文件内容，但反复
写入而不读回；2B 多次产生不符合动作协议的输出。不能把“低拒答率”当成自主执行、
持续学习或自我进化能力，也不能用四张图片推断模型的一般能力或认定 2B 劣于 0.8B。
本次没有证实失败来自审查，没有测试普通版与消融版的因果差异。

## Results

| Check | Huihui Qwen3.5 0.8B Q4_K_M | Huihui Qwen3.5 2B Q4_K_M |
|---|---|---|
| Image + valid-action probe | Passed | Passed |
| Synthetic localization, seed 42 | 1/4 hits; 4 valid clicks | 0/4 hits; 1 valid click, 3 action-validation errors |
| Median localization request time | 15.69 s | 51.48 s |
| Exact file + readback | Failed: correct bytes, 10 repeated writes, no readback | Failed: 2 errors, no dispatched action |
| Read/copy/launch trusted program | Failed: 2 errors, no child | Failed: 2 errors, no child |
| Real native desktop | Not tested | Not tested |
| Continual learning / recursive improvement | Not tested | Not tested |

A localization hit means the click falls inside the requested colored rectangle,
not exact centering. Four images are a smoke test, not a statistical benchmark.
Workbench success comes from independently checked files and child output, not a
model's `finish` declaration. Neither model launched a child during this run.
The copy task itself is only a bounded component test of a trusted print program;
it does not replicate Conway, install dependencies or recursively start agents.

The 0.8B file case used all 10 steps in 130.96 seconds. Its copy case stopped after
two consecutive errors in 22.52 seconds. The 2B cases stopped after two consecutive
errors in 67.01 and 64.22 seconds. Test-specific limits are deliberate; this does
not establish that every prompt, precision, runtime or larger model would fail.
No 4B model was tested here, and no automatic profile was replaced by these results.

## Method and provenance

- Conway 0.9.0, `conway-grounding-v2` and `conway-workbench-v1`.
- Linux x86_64, Intel Xeon Platinum 8573C, CPU inference, four threads; no GPU or native desktop.
- Python 3.12.14, Pillow 12.3.0. Other processes shared the host, so timing is descriptive.
- Official llama.cpp Ubuntu x64 build `b11146`, commit `7fe450e19`, version output `0.5.0-dev`.
- Model `Q4_K_M`, matching F16 multimodal projector, 8192 context tokens, one server slot,
  reasoning off, minimum image tokens 1024, local Web UI disabled.
- Generic JSON action adapter; temperature 0.1, at most 1000 output tokens per decision,
  HTTP request timeout 90 seconds. No model sampling seed was fixed.
- Four independent localization images with seed 42; the seed fixes images, not sampled responses.
- Two fresh workbench cases per model; synthetic screenshots but actual model decisions,
  actual temporary-file I/O, 10 steps / 180 seconds per case, two consecutive errors allowed.
- Time and stop checks are cooperative; an in-flight inference may return after the loop budget.
- All final cases are retained below, including failures. Raw arbitrary model prose is not logged
  in these reports; error types alone do not identify every underlying failure cause.

The weight files came from pinned community GGUF conversions of the publisher's
abliterated models. This is not a test of the publisher's full-precision weights.
The runtime archive was verified against the SHA256 in its GitHub release asset:
`c150306eb16b5ab696f76a8bdf810c35fd98a24e82158742e6fa28f420ff8410`.

Sources: [llama.cpp build b11146](https://github.com/ggml-org/llama.cpp/releases/tag/b11146),
[0.8B publisher card](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-0.8B-abliterated),
[2B publisher card](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-2B-abliterated).
Exact conversion revisions, file lengths and hashes are in each recorded report below.

## Prompt correction during investigation

An exploratory 0.8B run on grounding v1 returned the literal example coordinate
`(100, 100)` for all four images, with one incidental hit. A separate exploratory
workbench run also failed both tasks. This suggested example copying, but is not
a controlled causal result. We replaced literal tool examples with argument
definitions and versioned the changed localization prompt as v2. The final runs
above also explicitly set minimum image tokens to 1024, as recommended by this
runtime's startup warning for grounding. The final 0.8B result remained 1/4.
We did not select a successful retry or report the prompt change as an accuracy gain.

## Reproduce

Install Conway from the commit containing this report and the pinned llama.cpp
build. Download the two model files from one exact revision; for 0.8B:

```sh
hf download mradermacher/Huihui-Qwen3.5-0.8B-abliterated-GGUF Huihui-Qwen3.5-0.8B-abliterated.Q4_K_M.gguf Huihui-Qwen3.5-0.8B-abliterated.mmproj-f16.gguf --revision 2fabc82874616f44cdc494ec8ddc0e8ee10654b3 --local-dir ./acceptance-weights/0.8b
llama-server -m ./acceptance-weights/0.8b/Huihui-Qwen3.5-0.8B-abliterated.Q4_K_M.gguf --mmproj ./acceptance-weights/0.8b/Huihui-Qwen3.5-0.8B-abliterated.mmproj-f16.gguf --alias conway-huihui-0.8b --host 127.0.0.1 --port 8044 --ctx-size 8192 --parallel 1 --threads 4 --n-gpu-layers 0 --jinja --reasoning off --no-webui --image-min-tokens 1024
```

In another terminal, initialize a dedicated state directory, set `request_timeout: 90`
in its `config.yaml`, then run:

```sh
conway --home ./acceptance-state init
conway --home ./acceptance-state probe --endpoint http://127.0.0.1:8044/v1 --model conway-huihui-0.8b --brain generic
conway --home ./acceptance-state vision-check --endpoint http://127.0.0.1:8044/v1 --model conway-huihui-0.8b --brain generic --samples 4 --seed 42 --output ./vision-0.8b.json
conway --home ./acceptance-state acceptance-check --endpoint http://127.0.0.1:8044/v1 --model conway-huihui-0.8b --max-steps 10 --max-seconds 180 --output ./workbench-0.8b.json
```

For 2B, download the corresponding `2B` filenames from
`mradermacher/Huihui-Qwen3.5-2B-abliterated-GGUF` at revision
`f36848fead3fdda244cf60195c46993d23183d4c`; stop the first server and substitute
those paths, alias and output filenames. Verify weight hashes before use.
Reports require new output paths. Expect exit code 2 for failed task checks.
Image generation is reproducible with the pinned Pillow version; sampled responses
and timing are not guaranteed to match. The original run invoked the same Python
suite functions directly and supplemented their outputs with deployment provenance.

## Engineering checks

The release passed 230 local tests on Python 3.12, compilation and wheel building.
The wheel contains the new default goals and acceptance module. These automated
checks validate harness behavior, not model capability. The release commit's CI
provides separate Ubuntu, Windows and macOS fixture results.

For the architecture and future-model contract, see [FUTURE.md](../FUTURE.md).
There is currently no online weight update, automatic candidate promotion, Conway-trained
checkpoint or evidence that continual learning / recursive self-improvement has occurred.

## Recorded reports

The null deployment fields inside the generic vision output are supplemented by
its enclosing `provenance` object; they are not inferred by the evaluator.

### 0.8B

```json
{
  "protocol": {
    "status": "protocol_ok",
    "model": "conway-huihui-0.8b",
    "brain": "generic",
    "profile": "external",
    "image_request_accepted": true,
    "valid_action_type": "wait",
    "actions_executed": 0,
    "vision_grounding_verified": false,
    "note": "The endpoint accepted an image and returned a valid action. This is not a GUI-task or visual-accuracy benchmark."
  },
  "vision": {
    "schema_version": 1,
    "suite": "conway-grounding-v2",
    "conway_version": "0.9.0",
    "created_at": "2026-09-26T13:52:35.509967+00:00",
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
        "latency_seconds": 19.1839
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
        "latency_seconds": 16.1992
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
        "latency_seconds": 15.1896
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
          40,
          220
        ],
        "center_error_pixels": 515.425,
        "latency_seconds": 14.9038
      }
    ],
    "samples": 4,
    "hits": 1,
    "hit_rate": 0.25,
    "valid_clicks": 4,
    "errors": 0,
    "median_latency_seconds": 15.694400000000002,
    "actions_executed": 0,
    "real_desktop_tested": false,
    "note": "Synthetic color-target localization only. This does not measure real application task success."
  },
  "workbench": {
    "schema_version": 1,
    "suite": "conway-workbench-v1",
    "conway_version": "0.9.0",
    "created_at": "2026-09-26T13:55:08.999617+00:00",
    "model": "conway-huihui-0.8b",
    "brain": "generic",
    "profile": "external",
    "status": "failed",
    "cases": [
      {
        "case": "file_roundtrip",
        "loop_status": "stopped",
        "cycles": 10,
        "error_count": 0,
        "elapsed_seconds": 130.957,
        "actions": [
          "write_file",
          "write_file",
          "write_file",
          "write_file",
          "write_file",
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
        "loop_status": "error",
        "cycles": 2,
        "error_count": 2,
        "elapsed_seconds": 22.519,
        "actions": [],
        "errors": [
          "ValueError",
          "ActionValidationError"
        ],
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

### 2B

```json
{
  "protocol": {
    "status": "protocol_ok",
    "model": "conway-huihui-2b",
    "brain": "generic",
    "profile": "external",
    "image_request_accepted": true,
    "valid_action_type": "wait",
    "actions_executed": 0,
    "vision_grounding_verified": false,
    "note": "The endpoint accepted an image and returned a valid action. This is not a GUI-task or visual-accuracy benchmark."
  },
  "vision": {
    "schema_version": 1,
    "suite": "conway-grounding-v2",
    "conway_version": "0.9.0",
    "created_at": "2026-09-26T13:59:20.666014+00:00",
    "seed": 42,
    "model": "conway-huihui-2b",
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
        "valid_click": false,
        "error_type": "ActionValidationError",
        "latency_seconds": 56.7624
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
        "valid_click": false,
        "error_type": "ActionValidationError",
        "latency_seconds": 50.9941
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
        "hit": false,
        "valid_click": false,
        "error_type": "ActionValidationError",
        "latency_seconds": 43.1775
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
          700,
          250
        ],
        "center_error_pixels": 198.274,
        "latency_seconds": 51.9625
      }
    ],
    "samples": 4,
    "hits": 0,
    "hit_rate": 0.0,
    "valid_clicks": 1,
    "errors": 3,
    "median_latency_seconds": 51.478300000000004,
    "actions_executed": 0,
    "real_desktop_tested": false,
    "note": "Synthetic color-target localization only. This does not measure real application task success."
  },
  "workbench": {
    "schema_version": 1,
    "suite": "conway-workbench-v1",
    "conway_version": "0.9.0",
    "created_at": "2026-09-26T14:01:31.900904+00:00",
    "model": "conway-huihui-2b",
    "brain": "generic",
    "profile": "external",
    "status": "failed",
    "cases": [
      {
        "case": "file_roundtrip",
        "loop_status": "error",
        "cycles": 2,
        "error_count": 2,
        "elapsed_seconds": 67.005,
        "actions": [],
        "errors": [
          "ActionValidationError",
          "ValueError"
        ],
        "model_reported_completion": false,
        "passed": false,
        "checks": {
          "exact_file_content": false,
          "readback_performed": false
        },
        "output_sha256": null,
        "synthetic_observation": true,
        "real_model_expected": true,
        "child_launches": 0,
        "max_child_launches": 0
      },
      {
        "case": "bounded_replica",
        "loop_status": "error",
        "cycles": 2,
        "error_count": 2,
        "elapsed_seconds": 64.215,
        "actions": [],
        "errors": [
          "ActionValidationError",
          "JSONDecodeError"
        ],
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
  "provenance": {
    "repo": "mradermacher/Huihui-Qwen3.5-2B-abliterated-GGUF",
    "revision": "f36848fead3fdda244cf60195c46993d23183d4c",
    "quantization": "Q4_K_M",
    "projector": "F16",
    "runtime": "llama.cpp b11146 / 7fe450e19",
    "threads": 4,
    "context_size": 8192,
    "image_min_tokens": 1024,
    "reasoning": "off",
    "device": "CPU",
    "weights": {
      "Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf": {
        "bytes": 1270809024,
        "sha256": "aa25eea787afe56a097268f7ed3460cb623e1901d2e89cd2b654cabb42f80636"
      },
      "Huihui-Qwen3.5-2B-abliterated.mmproj-f16.gguf": {
        "bytes": 668227200,
        "sha256": "ca72dabcfb10c4954d533dd73e1365f790df0b9fe4941d9030b6a0d2c46abb7c"
      }
    }
  }
}
```
