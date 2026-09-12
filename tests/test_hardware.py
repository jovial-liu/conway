from conway.hardware import HardwareProfile, choose_model_profile


def test_model_selection_prefers_small_for_limited_memory():
    hardware = HardwareProfile("Linux", "x86_64", 16.0, "cpu", None, 12.0)
    name, profile = choose_model_profile(hardware)
    assert name == "small"
    assert "Qwen3-VL-4B" in profile["repo"]


def test_model_selection_prefers_standard_when_memory_allows():
    hardware = HardwareProfile("Linux", "x86_64", 32.0, "cuda", 24.0, 24.0)
    name, profile = choose_model_profile(hardware)
    assert name == "standard"
    assert "8B" in profile["repo"]
