from typing import List

import pytest

from maestro.pipelines.sam_segmentation import SamResponseProcessor


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("[1]", ["1"]),
        ("lorem ipsum [1] dolor sit amet", ["1"]),
        ("[1] lorem ipsum [2] dolor sit amet", ["1", "2"]),
        ("[1] lorem ipsum [1] dolor sit amet", ["1"]),
        ("[2] lorem ipsum [1] dolor sit amet", ["1", "2"])
    ]
)
def test_extract_marks_in_brackets(text: str, expected_result: List[str]) -> None:
    result = SamResponseProcessor.extract_mark_ids(text=text)
    assert result == expected_result
