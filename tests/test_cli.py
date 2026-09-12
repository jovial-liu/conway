from conway.cli import build_parser


def test_status_command_parses():
    args = build_parser().parse_args(["status"])
    assert args.command == "status"


def test_tiny_profile_parses():
    args = build_parser().parse_args(["start", "--profile", "tiny", "--mock", "--max-steps", "1"])
    assert args.profile == "tiny"
