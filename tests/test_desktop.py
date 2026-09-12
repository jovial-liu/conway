from conway.desktop import DesktopMetadata


def test_desktop_metadata_prompt_text():
    metadata = DesktopMetadata(active_app="Browser", active_window="Docs", provider="test")
    assert metadata.prompt_text() == "active app: Browser; active window: Docs"


def test_desktop_metadata_prompt_text_fallback():
    assert "unavailable" in DesktopMetadata().prompt_text()
