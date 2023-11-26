from typing import List

import pytest

from som.core import Mode
from som.utils import extract_marks_in_brackets


@pytest.mark.parametrize(
    "text, mode, expected_result",
    [
        ("[1]", Mode.NUMERIC, ["1"]),
        ("lorem ipsum [1] dolor sit amet", Mode.NUMERIC, ["1"]),
        ("[1] lorem ipsum [2] dolor sit amet", Mode.NUMERIC, ["1", "2"]),
        ("[A]", Mode.ALPHABETIC, ["A"]),
        ("lorem ipsum [A] dolor sit amet", Mode.ALPHABETIC, ["A"]),
        ("[A] lorem ipsum [B] dolor sit amet", Mode.ALPHABETIC, ["A", "B"]),
        ("[1] lorem ipsum [A] dolor sit amet", Mode.NUMERIC, ["1"]),
        ("[1] lorem ipsum [A] dolor sit amet", Mode.ALPHABETIC, ["A"]),
    ]
)
def test_extract_marks_in_brackets(
    text: str, mode: Mode,
    expected_result: List[str]
) -> None:
    result = extract_marks_in_brackets(text=text, mode=mode)
    assert result == expected_result
