"""Experimental reviewed-visual-policy LoRA recipe. GPU training is opt-in and unvalidated.

--validate-only checks dataset integrity without torch, model downloads, or training.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re

from conway.episodes import contained, digest, read_json
from conway.storage import atomic_write


def load_training_rows(root: Path) -> tuple[dict, dict[str, list]]:
    root = root.expanduser().resolve()
    manifest = read_json(root / 'dataset.json', limit=8_000_000)
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('files'), dict):
        raise ValueError('Unsupported training dataset manifest')
    for filename, expected in manifest['files'].items():
        path = contained(root, filename)
        if not path.is_file() or path.stat().st_size > 256_000_000 or digest(path.read_bytes()) != expected:
            raise ValueError('Training dataset integrity check failed')
    rows, ids, images = {}, {}, {}
    for split in ('train', 'validation'):
        filename = f'{split}.jsonl'
        if filename not in manifest['files']:
            raise ValueError('Dataset split is missing from the manifest')
        values = []
        with (root / filename).open(encoding='utf-8') as file:
            for line in file:
                if len(line) > 2_000_000:
                    raise ValueError('Training row exceeds 2 MB')
                item = json.loads(line)
                messages, paths, provenance = item['messages'], item['images'], item['provenance']
                if ([m.get('role') for m in messages] != ['system', 'user', 'assistant']
                        or len(paths) != 1 or paths[0] not in manifest['files']):
                    raise ValueError('Expected one image and system/user/assistant policy messages')
                image = contained(root, paths[0])
                episode_id, image_sha = provenance['episode_id'], provenance['image_sha256']
                if digest(image.read_bytes()) != image_sha:
                    raise ValueError('Image differs from provenance')
                if (episode_id in ids and ids[episode_id] != split) or (image_sha in images and images[image_sha] != split):
                    raise ValueError('Training/validation leakage detected')
                ids[episode_id], images[image_sha] = split, split
                # Completion-only loss: don't train the model to predict observations or tool descriptions.
                values.append({'prompt': messages[:-1], 'completion': messages[-1:], 'images': [str(image)]})
        rows[split] = values
        if len(values) != manifest['counts'][split]:
            raise ValueError('Dataset row count differs from manifest')
    return manifest, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', default='Qwen/Qwen3.5-0.8B')
    parser.add_argument('--revision', help='Immutable 40-character base-model Hub commit')
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    manifest, rows = load_training_rows(args.dataset)
    ready = bool(rows['train'] and rows['validation'])
    if args.validate_only:
        print(json.dumps({'status': 'dataset_valid', 'counts': manifest['counts'],
            'has_train_and_validation': ready, 'model_loaded': False, 'training_performed': False}, indent=2))
        return 0
    if not ready:
        raise ValueError('Training requires non-empty train and held-out validation episodes')
    if not args.revision or not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        raise ValueError('Training requires --revision with an immutable base-model commit')
    if args.output is None or args.output.exists() or not 1 <= args.max_steps <= 100000:
        raise ValueError('Training requires a new --output directory and 1..100000 max steps')
    import torch
    from datasets import Dataset, Image, Sequence
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    if not torch.cuda.is_available():
        raise RuntimeError('This experimental training recipe requires a CUDA GPU')
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    datasets = {k: Dataset.from_list(v).cast_column('images', Sequence(Image())) for k, v in rows.items()}
    args.output.mkdir(parents=True)
    training = {'status': 'starting', 'base_model': args.model, 'base_revision': args.revision,
        'dataset_manifest_sha256': digest((args.dataset/'dataset.json').read_bytes()),
        'dataset_counts': manifest['counts'], 'seed': args.seed, 'max_steps': args.max_steps,
        'method': 'LoRA SFT; completion-only loss; no online learning',
        'packages': {name: version(name) for name in ('torch', 'transformers', 'trl', 'peft', 'datasets', 'accelerate')}}
    atomic_write(args.output/'training.json', json.dumps(training, indent=2))
    try:
        trainer = SFTTrainer(model=args.model,
            args=SFTConfig(output_dir=str(args.output), max_steps=args.max_steps, max_length=None,
                per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
                learning_rate=2e-5, seed=args.seed, data_seed=args.seed,
                completion_only_loss=True, packing=False, report_to='none', push_to_hub=False,
                bf16=dtype == 'bfloat16', fp16=dtype == 'float16',
                model_init_kwargs={'revision': args.revision, 'dtype': dtype}),
            train_dataset=datasets['train'], eval_dataset=datasets['validation'],
            peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                   target_modules='all-linear', task_type='CAUSAL_LM'))
        trainer.train()
        metrics = trainer.evaluate()
        trainer.save_model(str(args.output/'adapter'))
        trainer.processing_class.save_pretrained(str(args.output/'adapter'))
        training.update(status='trained', validation_metrics=metrics,
            release_status='candidate adapter; no desktop evaluation or promotion performed')
    except BaseException:
        training['status'] = 'failed'
        raise
    finally:
        atomic_write(args.output/'training.json', json.dumps(training, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
