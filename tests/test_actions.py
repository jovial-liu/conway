from conway.actions import ActionValidationError, validate_action


def test_click_is_clamped_to_screen():
    action = validate_action({"type": "click", "args": {"x": 999, "y": -5}}, 100, 80)
    assert action == {"type": "click", "args": {"x": 99, "y": 0, "button": "left"}}


def test_hotkey_validation():
    action = validate_action({"type": "hotkey", "args": {"keys": ["CTRL", "L"]}}, 100, 80)
    assert action["args"]["keys"] == ["ctrl", "l"]


def test_rejects_unknown_action():
    try:
        validate_action({"type": "launch_missiles", "args": {}}, 100, 80)
    except ActionValidationError:
        pass
    else:
        raise AssertionError("unknown action was accepted")
