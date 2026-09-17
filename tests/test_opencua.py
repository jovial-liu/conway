from conway.opencua import model_point_to_screenshot, parse_opencua_action, smart_resize


def test_smart_resize_is_factor_aligned():
    height, width = smart_resize(1080, 1920)
    assert height % 28 == 0
    assert width % 28 == 0


def test_opencua_center_coordinate_maps_back_to_original():
    resized_h, resized_w = smart_resize(1080, 1920)
    x, y = model_point_to_screenshot(resized_w / 2, resized_h / 2, 1920, 1080)
    assert abs(x - 960) <= 1
    assert abs(y - 540) <= 1


def test_parse_opencua_click():
    resized_h, resized_w = smart_resize(1080, 1920)
    action = parse_opencua_action(
        f"pyautogui.click(x={resized_w / 2}, y={resized_h / 2})",
        1920,
        1080,
    )
    assert action["type"] == "click"
    assert abs(action["args"]["x"] - 960) <= 1
    assert abs(action["args"]["y"] - 540) <= 1


def test_parse_opencua_write_without_eval():
    action = parse_opencua_action("pyautogui.write('hello')", 100, 100)
    assert action == {"type": "type", "args": {"text": "hello", "interval": 0.01}}
