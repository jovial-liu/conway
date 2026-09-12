from conway.brain import OpenAICompatibleVLM


def test_extract_json_from_fenced_text():
    parsed = OpenAICompatibleVLM._extract_json('```json\n{"action":{"type":"wait","args":{}}}\n```')
    assert parsed["action"]["type"] == "wait"


def test_extract_json_from_surrounding_text():
    parsed = OpenAICompatibleVLM._extract_json('result: {"action":{"type":"wait","args":{}}}')
    assert parsed["action"]["type"] == "wait"
