import pytest

from conway.acceptance import SPECIMEN, WorkbenchExecutor, run_case
from conway.brain import Brain, Decision
from conway.computer import MockComputer


class FixturePolicy(Brain):
    def __init__(self, actions): self.actions = iter(actions)
    def decide(self, *args): return Decision(next(self.actions))


def test_independent_checks_do_not_accept_finish_as_success(tmp_path):
    report = run_case(FixturePolicy([{'type':'finish','args':{'reason':'claim'}}]), tmp_path,
                      'file_roundtrip', max_steps=1)
    assert report['model_reported_completion'] and not report['passed']


def test_file_workbench_checks_actual_content_and_readback(tmp_path):
    actions = [
        {'type':'write_file','args':{'path':'result.txt','content':'CONWAY_ACCEPTANCE_OK'}},
        {'type':'read_file','args':{'path':'result.txt'}},
        {'type':'finish','args':{'reason':'verified'}},
    ]
    report = run_case(FixturePolicy(actions), tmp_path, 'file_roundtrip')
    assert report['passed'] and report['cycles'] == 3 and report['child_launches'] == 0


def test_bounded_copy_launch_and_independent_output_check(tmp_path):
    actions = [
        {'type':'read_file','args':{'path':'seed.py'}},
        {'type':'write_file','args':{'path':'replica.py','content':SPECIMEN}},
        {'type':'shell','args':{'command':'python replica.py'}},
        {'type':'finish','args':{'reason':'verified'}},
    ]
    report = run_case(FixturePolicy(actions), tmp_path, 'bounded_replica')
    assert report['passed'] and report['child_launches'] == 1


def test_workbench_rejects_escape_and_arbitrary_program(tmp_path):
    executor = WorkbenchExecutor(MockComputer(tmp_path/'screens'), tmp_path, 'bounded_replica')
    observation = executor.computer.observe(1)
    for action in [
        {'type':'read_file','args':{'path':'../outside'}},
        {'type':'write_file','args':{'path':'seed.py','content':'change original'}},
        {'type':'shell','args':{'command':'python -c "print(1)"'}},
    ]:
        with pytest.raises(ValueError): executor.execute(action, observation)
    (tmp_path/'replica.py').write_text('print("untrusted substitute")')
    with pytest.raises(ValueError, match='trusted seed'):
        executor.execute({'type':'shell','args':{'command':'python replica.py'}}, observation)
