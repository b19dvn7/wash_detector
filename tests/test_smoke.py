from wash_detector.cli import main


def test_cli_smoke() -> None:
    assert callable(main)
