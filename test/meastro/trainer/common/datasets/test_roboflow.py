from contextlib import ExitStack

import pytest

from maestro.trainer.common.datasets.roboflow import parse_roboflow_identifier


@pytest.mark.parametrize(
    ("input_str", "expected_result", "context_manager"),
    [
        # 1. Full domain with protocol + version
        (
            "https://universe.roboflow.com/maestro/dataset/1",
            ("maestro", "dataset", 1),
            ExitStack(),
        ),
        # 2. App domain with protocol + version
        (
            "https://app.roboflow.com/maestro/dataset/1",
            ("maestro", "dataset", 1),
            ExitStack(),
        ),
        # 3. roboflow.com + version, no protocol
        (
            "roboflow.com/maestro/dataset/1",
            ("maestro", "dataset", 1),
            ExitStack(),
        ),
        # 4. No domain, just workspace/project + version
        (
            "maestro/dataset/1",
            ("maestro", "dataset", 1),
            ExitStack(),
        ),
        # 5. No version, just workspace/project
        (
            "maestro/dataset",
            ("maestro", "dataset", None),
            ExitStack(),
        ),
        # 6. Universe domain with protocol, no version
        (
            "https://universe.roboflow.com/maestro/dataset",
            ("maestro", "dataset", None),
            ExitStack(),
        ),
        # 7. App domain with protocol, no version
        (
            "https://app.roboflow.com/maestro/dataset",
            ("maestro", "dataset", None),
            ExitStack(),
        ),
        # 8. Universe domain (no protocol) + workspace/project
        (
            "universe.roboflow.com/maestro/dataset",
            ("maestro", "dataset", None),
            ExitStack(),
        ),
        # 9. App domain (no protocol) + workspace/project
        (
            "app.roboflow.com/maestro/dataset",
            ("maestro", "dataset", None),
            ExitStack(),
        ),
        # 10. Only one token -> invalid (return None)
        (
            "maestro",
            None,
            ExitStack(),
        ),
        # 11. Non-integer version -> invalid
        (
            "app.roboflow.com/maestro/dataset/version",
            None,
            ExitStack(),
        ),
        # 12. Extra tokens after version -> invalid
        (
            "app.roboflow.com/maestro/dataset/version/more",
            None,
            ExitStack(),
        ),
    ],
)
def test_parse_roboflow_identifier(input_str, expected_result, context_manager):
    with context_manager:
        result = parse_roboflow_identifier(input_str)
        assert result == expected_result
