"""Compare measured task outcomes; no inference, training or automatic promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def cases(report: dict) -> dict:
    if report.get('schema_version') != 1 or report.get('suite') != 'conway-workbench-v1':
        raise ValueError('Expected a conway-workbench-v1 report')
    rows = report.get('cases')
    if not isinstance(rows, list) or not rows:
        raise ValueError('A non-empty case list is required')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('case'), str):
            raise ValueError('Invalid case identity')
        name, checks = row['case'], row.get('checks')
        if not name or name in result:
            raise ValueError('Duplicate or empty case identity')
        if not isinstance(checks, dict) or not checks or any(type(v) is not bool for v in checks.values()):
            raise ValueError('Case checks must be a non-empty boolean mapping')
        if type(row.get('passed')) is not bool or row['passed'] != all(checks.values()):
            raise ValueError('Case pass label disagrees with independent checks')
        result[name] = row
    if report.get('status') != ('passed' if all(r['passed'] for r in rows) else 'failed'):
        raise ValueError('Suite status disagrees with case results')
    return result


def compare(baseline: dict, candidate: dict) -> dict:
    before, after = cases(baseline), cases(candidate)
    budgets_known = isinstance(baseline.get('budgets'), dict) and isinstance(candidate.get('budgets'), dict)
    if budgets_known and baseline['budgets'] != candidate['budgets']:
        raise ValueError('Run budgets differ; use matched budgets for comparison')
    if before.keys() != after.keys():
        raise ValueError('Task sets differ; missing tasks cannot count as improvement')
    for name in before:
        if before[name]['checks'].keys() != after[name]['checks'].keys():
            raise ValueError(f'Check definitions differ for {name}')
    gained = sorted(k for k in before if not before[k]['passed'] and after[k]['passed'])
    lost = sorted(k for k in before if before[k]['passed'] and not after[k]['passed'])
    return {'schema_version': 1, 'suite': baseline['suite'],
            'status': 'regression' if lost else 'observed_gain' if gained else 'no_observed_gain',
            'cases': len(before), 'baseline_passed': sum(r['passed'] for r in before.values()),
            'candidate_passed': sum(r['passed'] for r in after.values()),
            'gained': gained, 'lost': lost,
            'candidate_passed_all': all(r['passed'] for r in after.values()),
            'matching_budget_metadata': True if budgets_known else None,
            'baseline': {k: baseline.get(k) for k in ('model', 'conway_version', 'created_at')},
            'candidate': {k: candidate.get(k) for k in ('model', 'conway_version', 'created_at')},
            'automatic_promotion': False,
            'note': 'Descriptive comparison of supplied reports, not an authenticated evaluator or proof of learning. '
                    'Control weights, tasks, budgets and runtime; repeat on held-out tasks before drawing conclusions.'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args(argv)
    try:
        raw = []
        for path in (args.baseline, args.candidate):
            with path.open('rb') as file:
                data = file.read(2_000_001)
            if len(data) > 2_000_000:
                raise ValueError('Report exceeds 2 MB')
            raw.append(data)
        reports = [json.loads(data) for data in raw]
        if any(not isinstance(r, dict) for r in reports):
            raise ValueError('Reports must be JSON objects')
        result = compare(*reports)
        result['input_sha256'] = {name: hashlib.sha256(data).hexdigest()
                                  for name, data in zip(('baseline', 'candidate'), raw)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result['lost'] else 0
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Cannot compare: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
