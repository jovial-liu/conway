from pathlib import Path
import json

from conway.memory import FileMemory


def test_file_memory_keeps_notes_and_recent_events(tmp_path: Path):
    memory = FileMemory(tmp_path)
    memory.append_memory("durable fact")
    memory.journal({"cycle": 1, "result": "ok"})
    context = memory.recall()
    assert "durable fact" in context
    assert '"cycle": 1' in context


def test_memory_can_be_replaced_by_rolling_summary(tmp_path: Path):
    memory = FileMemory(tmp_path)
    memory.memory_path.write_text("x" * 200, encoding="utf-8")
    assert memory.needs_compaction(max_bytes=100)
    memory.replace_memory("## Active work\nContinue task A")
    text = memory.memory_path.read_text(encoding="utf-8")
    assert "Continue task A" in text
    assert not memory.needs_compaction(max_bytes=100)


def test_recent_execution_is_visible_without_old_visual_noise(tmp_path):
    memory = FileMemory(tmp_path)
    memory.append_memory('Keep the useful older lesson')
    memory.journal({'event':'action_result', 'cycle':1, 'dry_run':False,
        'action':{'type':'write_file','args':{'path':'result.txt','content':'correct'}},
        'result':'wrote 7 characters', 'observation':{'ui_tree':'visual noise '*2000},
        'rationale':'model summary', 'memory_note':'unverified claim'})
    context = memory.recall(2000)
    assert context.index('write_file') < context.index('Keep the useful older lesson')
    assert 'wrote 7 characters' in context and 'result.txt' in context
    assert 'visual noise' not in context and 'unverified claim' not in context
    assert len(context) <= 2000
    assert 'visual noise' in memory.recent_events(max_chars=40000)


def test_latest_error_and_dry_run_remain_distinct_from_execution(tmp_path):
    memory = FileMemory(tmp_path)
    memory.journal({'event':'action_result','cycle':1,'dry_run':True,'result':'planned only'})
    assert '"dry_run": true' in memory.recall()
    memory.journal({'event':'cycle_error','cycle':2,'error':'invalid args.x',
                    'action_outcome_uncertain':False})
    memory.journal({'event':'memory_compaction','cycle':2,'after_bytes':300})
    context = memory.recall()
    assert context.index('invalid args.x') < context.index('planned only')
    assert '"action_outcome_uncertain": false' in context
    for line in memory.recent_events(max_chars=1200, compact=True).splitlines():
        assert isinstance(json.loads(line),dict)
