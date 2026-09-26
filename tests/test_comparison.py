from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('compare_acceptance',
    Path(__file__).resolve().parents[1]/'scripts'/'compare_acceptance.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report(a, b):
    return {'schema_version':1, 'suite':'conway-workbench-v1',
            'status':'passed' if a and b else 'failed',
            'cases':[{'case':name, 'checks':{'artifact':value}, 'passed':value}
                     for name, value in (('old_skill',a), ('new_skill',b))]}


def test_new_gain_does_not_hide_a_lost_capability():
    result = module.compare(report(True,False),report(False,True))
    assert result['status'] == 'regression'
    assert result['gained'] == ['new_skill'] and result['lost'] == ['old_skill']
    assert result['automatic_promotion'] is False


def test_removed_task_or_check_is_not_a_valid_improvement():
    baseline = report(False,False)
    changed = deepcopy(baseline)
    changed['cases'].pop()
    with pytest.raises(ValueError,match='Task sets differ'):
        module.compare(baseline,changed)
    changed = deepcopy(baseline)
    changed['cases'][0]['checks'] = {'different_check':False}
    with pytest.raises(ValueError,match='Check definitions differ'):
        module.compare(baseline,changed)


def test_self_reported_success_cannot_override_failed_checks():
    baseline = report(False,False)
    candidate = deepcopy(baseline)
    candidate['cases'][0]['passed'] = True
    with pytest.raises(ValueError,match='pass label disagrees'):
        module.compare(baseline,candidate)


def test_unchanged_score_is_not_an_observed_gain():
    assert module.compare(report(True,False),report(True,False))['status'] == 'no_observed_gain'
    result = module.compare(report(True,False),report(True,True))
    assert result['candidate_passed_all'] and result['status'] == 'observed_gain'


def test_extra_compute_is_not_silently_treated_as_matched_evidence():
    baseline, candidate = report(False,False), report(True,True)
    assert module.compare(baseline,candidate)['matching_budget_metadata'] is None
    baseline['budgets'] = {'max_steps':10, 'max_seconds':180, 'max_consecutive_errors':2}
    candidate['budgets'] = {'max_steps':100, 'max_seconds':180, 'max_consecutive_errors':2}
    with pytest.raises(ValueError,match='budgets differ'):
        module.compare(baseline,candidate)
