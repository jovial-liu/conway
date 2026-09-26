"""Seeded synthetic visual grounding checks. No native desktop or action executor."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import random
import statistics
from tempfile import TemporaryDirectory
import time

from PIL import Image, ImageDraw
from . import __version__
from .actions import validate_action
from .computer import Observation
from .config import ConwayConfig
from .diagnostics import model_session

SUITE_VERSION = 'conway-grounding-v1'
COLORS = {'red': '#D92D20', 'blue': '#175CD3', 'green': '#067647', 'purple': '#7A3EB1'}


@dataclass(frozen=True)
class GroundingCase:
    case_id: int
    instruction: str
    bounds: tuple[int, int, int, int]
    image_sha256: str
    observation: Observation


def make_cases(directory: Path, *, samples: int = 8, seed: int = 0) -> list[GroundingCase]:
    if type(samples) is not int or not 1 <= samples <= 100:
        raise ValueError('samples must be an integer from 1 to 100')
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError('seed must be an integer from 0 to 4294967295')
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    cases = []
    for index in range(samples):
        width, height = ((840, 560), (1120, 700), (700, 840))[index % 3]
        image = Image.new('RGB', (width, height), '#F4F6F9')
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width - 1, 44), fill='#202939')
        draw.text((18, 16), 'Conway synthetic visual test', fill='white')
        names = list(COLORS)
        rng.shuffle(names)
        target = rng.choice(names)
        target_bounds = None
        # Each color appears once, with randomized position and size in four disjoint cells.
        for cell, color in enumerate(names):
            col, row = cell % 2, cell // 2
            left = col * width // 2 + rng.randint(24, 70)
            top = 65 + row * (height - 65) // 2 + rng.randint(12, 35)
            right = left + rng.randint(width // 8, width // 4)
            bottom = top + rng.randint(50, 90)
            box = (left, top, right, bottom)
            draw.rectangle(box, fill=COLORS[color])
            if color == target:
                target_bounds = box
        path = directory / f'case-{index + 1:03d}.png'
        image.save(path)
        observation = Observation(path, width, height, width, height, metadata_provider='synthetic')
        cases.append(GroundingCase(index + 1,
            f'Click the center of the {target} rectangle. Return one left click. This is a synthetic test image.',
            target_bounds, hashlib.sha256(path.read_bytes()).hexdigest(), observation))
    return cases


def run_grounding(brain, cases: list[GroundingCase]) -> dict:
    if not cases:
        raise ValueError('At least one grounding case is required')
    results = []
    for case in cases:
        start = time.monotonic()
        row = {'case_id': case.case_id, 'instruction': case.instruction,
               'image_sha256': case.image_sha256,
               'image_size': [case.observation.width, case.observation.height],
               'target_bounds': list(case.bounds), 'hit': False, 'valid_click': False}
        try:
            # Coordinates/answers are never supplied as text or trajectory context to the model.
            decision = brain.decide(case.instruction, 'No previous actions. Use only the current image.', case.observation)
            action = validate_action(decision.action, case.observation.width, case.observation.height)
            row['action_type'] = action['type']
            if action['type'] == 'click' and action['args']['button'] == 'left':
                x, y = action['args']['x'], action['args']['y']
                left, top, right, bottom = case.bounds
                row.update(valid_click=True, point=[x, y], hit=left <= x <= right and top <= y <= bottom,
                           center_error_pixels=round(math.hypot(x - (left + right) / 2, y - (top + bottom) / 2), 3))
        except Exception as exc:
            # Do not publish endpoint error bodies, credentials, or model-generated arbitrary text.
            row['error_type'] = type(exc).__name__
        row['latency_seconds'] = round(time.monotonic() - start, 4)
        results.append(row)
    hits = sum(row['hit'] for row in results)
    return {'cases': results, 'samples': len(results), 'hits': hits, 'hit_rate': hits / len(results),
            'valid_clicks': sum(row['valid_click'] for row in results),
            'errors': sum('error_type' in row for row in results),
            'median_latency_seconds': statistics.median(row['latency_seconds'] for row in results),
            'actions_executed': 0, 'real_desktop_tested': False,
            'note': 'Synthetic color-target localization only. This does not measure real application task success.'}


def vision_check(config: ConwayConfig, state_root: Path, *, samples: int = 8, seed: int = 0) -> dict:
    with TemporaryDirectory(prefix='conway-vision-') as directory:
        cases = make_cases(Path(directory), samples=samples, seed=seed)
        with model_session(config, state_root,
                           tool_manifest='- click: {"x": 100, "y": 100, "button": "left"}') as (brain, metadata):
            result = run_grounding(brain, cases)
    return {'schema_version': 1, 'suite': SUITE_VERSION, 'conway_version': __version__,
            'created_at': datetime.now(timezone.utc).isoformat(), 'seed': seed, **metadata,
            'processor': {'opencua_min_pixels': config.opencua_min_pixels,
                          'opencua_max_pixels': config.opencua_max_pixels},
            'model_revision': None, 'quantization': None, 'inference_engine_version': None,
            'status': 'passed' if result['hits'] == samples else 'failed', **result}
