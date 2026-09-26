"""Opt-in local visual rollouts and reviewed policy datasets. No upload or self-labeling."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

from PIL import Image
from . import __version__
from .actions import validate_action
from .brain import CONTEXT_BOUNDARY, decision_instruction
from .storage import atomic_write


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path, limit=2_000_000):
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f'Invalid or oversized dataset file: {path.name}')
    return json.loads(path.read_text(encoding='utf-8'))


def contained(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError('Dataset paths must be relative')
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Dataset path escapes its directory')
    return path


class EpisodeRecorder:
    def __init__(self, root: Path, *, model: str, brain: str, dry_run: bool, synthetic: bool,
                 tool_manifest: str, state_root: Path) -> None:
        self.root = root.expanduser().resolve()
        if self.root.exists() or self.root.is_relative_to(state_root.resolve()):
            raise ValueError('Episode directory must be new and outside the state directory')
        self.root.mkdir(parents=True, mode=0o700)
        self.manifest = tool_manifest
        self.bytes_written = 0
        self.meta = {'schema_version': 1, 'episode_id': uuid.uuid4().hex, 'conway_version': __version__,
            'model': model, 'brain': brain, 'dry_run': dry_run, 'synthetic': synthetic,
            'source': 'policy_rollout', 'created_at': datetime.now(timezone.utc).isoformat(),
            'status': 'recording', 'steps': []}
        self._save()

    def _save(self):
        atomic_write(self.root / 'episode.json', json.dumps(self.meta, ensure_ascii=False, indent=2))

    def record(self, *, step_id, cycle, constitution, memory, observation, decision, action, result):
        # Protect disk use during unattended recording. Start a new bounded session for more samples.
        if len(self.meta['steps']) >= 1000:
            raise ValueError('Episode reached 1000 steps; start a new bounded recording')
        image = observation.screenshot_path.read_bytes()
        if len(image) > 20_000_000:
            raise ValueError('Episode screenshot exceeds 20 MB')
        if self.bytes_written + len(image) > 256_000_000:
            raise ValueError('Episode screenshot budget reached 256 MB')
        image_name = f'{step_id}.png'
        path = self.root / image_name
        with path.open('xb') as out:
            out.write(image)
        path.chmod(0o600)
        self.bytes_written += len(image)
        record = {'schema_version': 1, 'episode_id': self.meta['episode_id'], 'step_id': step_id,
            'cycle': cycle, 'image': image_name, 'image_sha256': digest(image),
            'width': observation.width, 'height': observation.height,
            'system': constitution + CONTEXT_BOUNDARY,
            'instruction': decision_instruction(memory, observation, self.manifest),
            'action': action, 'result': str(result), 'task_success': None,
            'response': {'action': action, 'rationale': decision.rationale, 'memory_note': decision.memory_note}}
        filename = f'{step_id}.json'
        atomic_write(self.root / filename, json.dumps(record, ensure_ascii=False))
        self.meta['steps'].append({'path': filename, 'sha256': digest((self.root / filename).read_bytes())})
        self._save()

    def close(self, status: str):
        self.meta['status'] = status
        self._save()


def reviewed_steps(root: Path):
    root = root.expanduser().resolve()
    episode = read_json(root / 'episode.json')
    if (episode.get('schema_version') != 1 or episode.get('dry_run') is not False
            or episode.get('synthetic') is not False or episode.get('brain') != 'generic'
            or episode.get('status') == 'recording'):
        raise ValueError('Training export requires a closed, real, executed generic-policy episode')
    review = read_json(root / 'review.json')
    if (review.get('schema_version') != 1 or review.get('episode_id') != episode.get('episode_id')
            or review.get('source') not in {'human', 'external_evaluator'}
            or not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip()
            or review.get('privacy_reviewed') is not True):
        raise ValueError('Episode requires an independent review with privacy_reviewed=true')
    labels = review.get('steps')
    if not isinstance(labels, dict):
        raise ValueError('Review steps must be a mapping')
    seen = set()
    for item in episode['steps']:
        path = contained(root, item['path'])
        step = read_json(path)
        if digest(path.read_bytes()) != item['sha256']:
            raise ValueError('Recorded step hash mismatch')
        step_id = step.get('step_id')
        if not isinstance(step_id, str) or step_id in seen or step.get('episode_id') != episode['episode_id']:
            raise ValueError('Invalid or duplicate step identity')
        seen.add(step_id)
        label = labels.get(step_id, {})
        if not isinstance(label, dict):
            raise ValueError('Step review must be an object')
        if label.get('accepted') is not True:
            continue
        if label.get('outcome') != 'passed':
            raise ValueError('Accepted step requires a passed external outcome')
        evidence = contained(root, label.get('evidence'))
        if not evidence.is_file() or not 1 <= evidence.stat().st_size <= 2_000_000:
            raise ValueError('Accepted step needs a non-empty evidence file up to 2 MB')
        image = contained(root, step['image'])
        if image.stat().st_size > 20_000_000 or digest(image.read_bytes()) != step['image_sha256']:
            raise ValueError('Recorded screenshot hash mismatch')
        with Image.open(image) as picture:
            if picture.format != 'PNG' or picture.size != (step['width'], step['height']):
                raise ValueError('Screenshot dimensions/format differ from action coordinates')
            picture.verify()
        if validate_action(step['action'], step['width'], step['height']) != step['action']:
            raise ValueError('Training action is not canonical')
        if step['response'].get('action') != step['action']:
            raise ValueError('Response/action mismatch')
        yield episode, step, image, {'source': review['source'], 'reviewer': review['reviewer'],
                                    'evidence_sha256': digest(evidence.read_bytes())}
    if set(labels) - seen:
        raise ValueError('Review refers to an unknown step')


def export_dataset(episodes: list[Path], output: Path, validation_percent: int = 20) -> dict:
    if type(validation_percent) is not int or not 0 <= validation_percent <= 99:
        raise ValueError('validation_percent must be 0..99')
    output = output.expanduser().resolve()
    if output.exists() or any(output.is_relative_to(p.expanduser().resolve()) for p in episodes):
        raise ValueError('Dataset output must be a new directory outside the input episodes')
    output.mkdir(parents=True, mode=0o700)
    counts, identities, hashes = {'train': 0, 'validation': 0}, set(), {}
    try:
        (output / 'images').mkdir()
        for root in episodes:
            meta = read_json(root.expanduser().resolve() / 'episode.json')
            identity = meta.get('episode_id')
            if not isinstance(identity, str) or identity in identities:
                raise ValueError('Episode IDs must be unique strings')
            identities.add(identity)
            split = 'validation' if int(digest(identity.encode())[:8], 16) % 100 < validation_percent else 'train'
            for episode, step, image, review in reviewed_steps(root):
                # Prevent screenshot-identical examples from leaking across train/evaluation.
                image_hash = step['image_sha256']
                if image_hash in hashes and hashes[image_hash] != split:
                    raise ValueError('An identical screenshot occurs in both dataset splits')
                hashes[image_hash] = split
                relative = f'images/{image_hash}.png'
                if not (output / relative).exists():
                    shutil.copyfile(image, output / relative)
                    (output / relative).chmod(0o600)
                row = {'messages': [
                    {'role': 'system', 'content': [{'type': 'text', 'text': step['system']}]},
                    {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': step['instruction']}]},
                    {'role': 'assistant', 'content': [{'type': 'text', 'text': json.dumps(step['response'], ensure_ascii=False)}]}],
                    'images': [relative], 'provenance': {'episode_id': episode['episode_id'], 'step_id': step['step_id'],
                    'model': episode['model'], 'image_sha256': image_hash, 'review': review}}
                with (output / f'{split}.jsonl').open('a', encoding='utf-8') as file:
                    file.write(json.dumps(row, ensure_ascii=False) + '\n')
        # Counts are computed from committed rows, rather than inferred from labels.
        for split in counts:
            path = output / f'{split}.jsonl'
            if not path.exists():
                path.touch(mode=0o600)
            path.chmod(0o600)
            counts[split] = sum(1 for _ in path.open(encoding='utf-8'))
        if not sum(counts.values()):
            raise ValueError('No reviewed steps eligible for training')
        report = {'schema_version': 1, 'counts': counts, 'episodes': sorted(identities),
                  'split_method': 'sha256 episode ID modulo 100', 'validation_percent': validation_percent,
                  'review_semantics': 'operator-supplied annotations; evidence is not automatically fact-checked',
                  'files': {p.relative_to(output).as_posix(): digest(p.read_bytes())
                            for p in sorted(output.rglob('*')) if p.is_file()}}
        atomic_write(output / 'dataset.json', json.dumps(report, indent=2))
        return report
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        shutil.rmtree(output)
        raise ValueError('Malformed episode or review data') from exc
    except BaseException:
        shutil.rmtree(output)
        raise
