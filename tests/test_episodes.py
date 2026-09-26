import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from conway.actions import validate_action
from conway.brain import Decision, OpenAICompatibleVLM
from conway.cli import main
from conway.computer import Observation
from conway.episodes import EpisodeRecorder, digest, export_dataset


def episode(tmp_path, name='episode', *, dry_run=False, synthetic=False, brain='generic', review=True, color='green'):
    image = tmp_path / f'{name}.png'
    Image.new('RGB', (16, 16), color).save(image)
    obs = Observation(image, 16, 16, 16, 16)
    recorder = EpisodeRecorder(tmp_path / name, model='test-fixture-policy', brain=brain,
        dry_run=dry_run, synthetic=synthetic, tool_manifest='- wait: {"seconds":1}', state_root=tmp_path/'state')
    action = validate_action({'type': 'wait', 'args': {'seconds': 0}}, 16, 16)
    recorder.record(step_id='step1', cycle=1, constitution='Fixture only', memory='prior observation',
        observation=obs, decision=Decision(action, 'fixture summary'), action=action, result='waited')
    recorder.close('stopped')
    root = recorder.root
    (root / 'evidence.txt').write_text('TEST FIXTURE ONLY: output independently inspected.', encoding='utf-8')
    if review:
        (root / 'review.json').write_text(json.dumps({'schema_version': 1, 'episode_id': recorder.meta['episode_id'],
            'source': 'external_evaluator', 'reviewer': 'test-fixture', 'privacy_reviewed': True,
            'steps': {'step1': {'accepted': True, 'outcome': 'passed', 'evidence': 'evidence.txt'}}}), encoding='utf-8')
    return root


def test_reviewed_visual_data_roundtrip_and_prompt_alignment(tmp_path):
    root = episode(tmp_path)
    report = export_dataset([root], tmp_path / 'dataset', 0)
    assert report['counts'] == {'train': 1, 'validation': 0}
    row = json.loads((tmp_path / 'dataset/train.jsonl').read_text())
    assert row['messages'][1]['content'][0] == {'type': 'image'}
    assert (tmp_path / 'dataset' / row['images'][0]).is_file()
    assert row['provenance']['review']['source'] == 'external_evaluator'
    stored = json.loads((root / 'step1.json').read_text())
    assert stored['task_success'] is None
    import httpx
    def response(request):
        payload = json.loads(request.content)
        assert payload['messages'][0]['content'] == stored['system']
        assert payload['messages'][1]['content'][1]['text'] == stored['instruction']
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(stored['response'])}}]})
    brain = OpenAICompatibleVLM(tool_manifest='- wait: {"seconds":1}', transport=httpx.MockTransport(response))
    try:
        brain.decide('Fixture only', 'prior observation', Observation(tmp_path/'episode.png',16,16,16,16))
    finally:
        brain.close()


@pytest.mark.parametrize('kwargs', [{'dry_run': True}, {'synthetic': True}, {'brain': 'opencua'}, {'review': False}])
def test_non_training_episodes_rejected(tmp_path, kwargs):
    root = episode(tmp_path, **kwargs)
    with pytest.raises((ValueError, OSError)):
        export_dataset([root], tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('change', ['image', 'step', 'self_review', 'privacy', 'escape', 'unknown'])
def test_corruption_and_incomplete_review_rejected(tmp_path, change):
    root = episode(tmp_path)
    review = json.loads((root / 'review.json').read_text())
    if change == 'image':
        (root / 'step1.png').write_bytes(b'corrupt')
    elif change == 'step':
        (root / 'step1.json').write_text('{}')
    elif change == 'self_review':
        review['source'] = 'model_self_report'
    elif change == 'privacy':
        review['privacy_reviewed'] = False
    elif change == 'escape':
        review['steps']['step1']['evidence'] = '../private.txt'
    else:
        review['steps']['unknown'] = {'accepted': False}
    (root / 'review.json').write_text(json.dumps(review))
    with pytest.raises(ValueError):
        export_dataset([root], tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_duplicate_episode_and_cross_split_screenshot_rejected(tmp_path):
    one, two = episode(tmp_path, 'one'), episode(tmp_path, 'two')
    with pytest.raises(ValueError, match='unique'):
        export_dataset([one, one], tmp_path / 'out')
    meta_path = two / 'episode.json'
    # Same image in two distinct episodes, intentionally force opposite splits.
    first_id = json.loads((one/'episode.json').read_text())['episode_id']
    first_bucket = int(digest(first_id.encode())[:8], 16) % 100 < 50
    new_id = next(str(n) for n in range(100) if (int(digest(str(n).encode())[:8],16)%100 < 50) != first_bucket)
    meta, step, review = [json.loads((two/f).read_text()) for f in ('episode.json','step1.json','review.json')]
    for item in (meta, step, review):
        item['episode_id'] = new_id
    (two / 'step1.json').write_text(json.dumps(step))
    meta['steps'][0]['sha256'] = digest((two/'step1.json').read_bytes())
    meta_path.write_text(json.dumps(meta))
    (two/'review.json').write_text(json.dumps(review))
    with pytest.raises(ValueError, match='both dataset splits'):
        export_dataset([one, two], tmp_path / 'out', 50)


def test_mock_cli_records_without_stdin_but_cannot_be_training_data(tmp_path):
    assert main(['--home', str(tmp_path/'state'), 'start', '--mock', '--max-steps', '1',
                 '--record-episode', str(tmp_path/'recording'), '--quiet']) == 0
    metadata = json.loads((tmp_path/'recording/episode.json').read_text())
    assert metadata['synthetic'] is True and len(metadata['steps']) == 1
    assert metadata['status'] != 'recording'
    with pytest.raises(ValueError, match='real, executed'):
        export_dataset([tmp_path/'recording'], tmp_path/'out')


def test_episode_does_not_overwrite_or_live_inside_state(tmp_path):
    with pytest.raises(ValueError, match='new and outside'):
        EpisodeRecorder(tmp_path/'state/recording', model='x', brain='generic', dry_run=False, synthetic=False,
                        tool_manifest='', state_root=tmp_path/'state')


def test_training_entrypoint_validates_without_loading_model(tmp_path, capsys):
    root = episode(tmp_path)
    export_dataset([root], tmp_path/'dataset', 0)
    spec = importlib.util.spec_from_file_location('train_policy', Path(__file__).resolve().parents[1]/'scripts/train_policy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(['--dataset', str(tmp_path/'dataset'), '--validate-only']) == 0
    assert json.loads(capsys.readouterr().out)['training_performed'] is False
    _, rows = module.load_training_rows(tmp_path/'dataset')
    assert len(rows['train'][0]['prompt']) == 2 and len(rows['train'][0]['completion']) == 1
    (tmp_path/'dataset/train.jsonl').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        module.load_training_rows(tmp_path/'dataset')
