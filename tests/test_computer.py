from pathlib import Path

from conway.computer import Computer, Observation


def test_maps_retina_screenshot_coordinates_to_input_coordinates():
    observation = Observation(
        screenshot_path=Path("screen.png"),
        width=2000,
        height=1200,
        input_width=1000,
        input_height=600,
    )
    assert Computer.map_screenshot_point(1000, 600, observation) == (500, 300)


def test_coordinate_mapping_is_clamped_to_input_space():
    observation = Observation(
        screenshot_path=Path("screen.png"),
        width=2000,
        height=1200,
        input_width=1000,
        input_height=600,
    )
    assert Computer.map_screenshot_point(1999, 1199, observation) == (999, 599)
