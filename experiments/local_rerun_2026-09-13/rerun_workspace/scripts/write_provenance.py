#!/usr/bin/env python3
"""Write reproducibility manifests for the local rerun delivery."""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rerun_workspace" / "manifest"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add(rows: list[dict], kind: str, label: str, path: str | Path, url: str = "", notes: str = "") -> None:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    rows.append({
        "kind": kind,
        "label": label,
        "path": str(candidate),
        "exists": candidate.is_file(),
        "sha256": sha256(candidate) if candidate.is_file() else "",
        "source_url": url,
        "notes": notes,
    })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    add(rows, "script", "local inference runner", "rerun_workspace/scripts/run_local_rerun.py")
    add(rows, "script", "local statistical analysis", "rerun_workspace/scripts/analyze_local_rerun.py")
    add(rows, "script", "output validation/provenance writer", "rerun_workspace/scripts/write_provenance.py")
    add(rows, "script", "output validation", "rerun_workspace/scripts/validate_outputs.py")
    add(rows, "config", "local rerun configuration", "rerun_workspace/config/local_rerun_config.json")
    add(rows, "protocol", "archived protocol", "paper_evidence_complete_2026-09-12/evidence/recovered/16_PROTOCOL_FOR_MANUSCRIPT.md")
    add(rows, "source", "archived selector definitions", "recovery_work/source/run_reviewer_identification_round2.py")
    add(rows, "source", "archived multisetting recovery", "recovery_work/source/run_cci_preserving_multisetting.py")
    add(rows, "frozen", "accepted frozen experiment package", "experiment_results.zip", notes="Read-only reference; not overwritten")
    add(rows, "model", "OpenAI CLIP ViT-B/16", "rerun_workspace/model_weights/ViT-B-16.pt", "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt")
    add(rows, "model", "OpenAI CLIP ViT-B/32", "rerun_workspace/model_weights/ViT-B-32.pt", "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt")
    add(rows, "dataset_archive", "COCO val2014 images", "/tmp/coco_val2014.zip", "http://images.cocodataset.org/zips/val2014.zip")
    add(rows, "dataset_archive", "COCO trainval annotations", "/tmp/annotations_trainval2014.zip", "https://images.cocodataset.org/annotations/annotations_trainval2014.zip")
    add(rows, "dataset_file", "COCO validation annotations used", "rerun_workspace/data/coco/annotations/instances_val2014.json", "https://images.cocodataset.org/annotations/annotations_trainval2014.zip", "Extracted input; archive may not be retained")
    add(rows, "dataset_archive", "VOC2007 trainval", "/tmp/VOCtrainval_06-Nov-2007.tar", "https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar")
    add(rows, "dataset_archive", "VOC2007 test", "/tmp/VOCtest_06-Nov-2007.tar", "https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar")
    with (OUT / "source_and_hashes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": importlib.import_module("torch").__version__,
        "mps_built": bool(importlib.import_module("torch").backends.mps.is_built()),
        "mps_available": bool(importlib.import_module("torch").backends.mps.is_available()),
    }
    (OUT / "dependency_versions.txt").write_text(json.dumps(env, indent=2) + "\n\n" + freeze)


if __name__ == "__main__":
    main()
