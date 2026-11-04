from typing import Optional
from unittest import mock

import pytest
from typer.testing import CliRunner

from maestro.cli.main import app

TRAIN_COMMANDS = [
    ("florence_2", "maestro.trainer.models.florence_2.entrypoint", "florence_2_train"),
    ("paligemma_2", "maestro.trainer.models.paligemma_2.entrypoint", "paligemma_2_train"),
    ("qwen_2_5_vl", "maestro.trainer.models.qwen_2_5_vl.entrypoint", "qwen_2_5_vl_train"),
]


@pytest.mark.parametrize(("model_name", "entrypoint_path", "train_name"), TRAIN_COMMANDS)
@pytest.mark.parametrize(
    ("extra_args", "expected_peft_advanced_params"),
    [
        # 1. Flag omitted -> default LoRA config is requested
        ([], None),
        # 2. Flag passed -> JSON is parsed into a dictionary
        (["--peft_advanced_params", '{"r": 16}'], {"r": 16}),
    ],
)
def test_train_command_passes_configuration_to_train(
    model_name: str,
    entrypoint_path: str,
    train_name: str,
    extra_args: list[str],
    expected_peft_advanced_params: Optional[dict[str, int]],
) -> None:
    # Model recipes need their optional extras, which a base install does not have.
    entrypoint = pytest.importorskip(entrypoint_path)

    with mock.patch.object(entrypoint, train_name) as train_mock:
        result = CliRunner().invoke(app, [model_name, "train", "--dataset", "some/dataset", *extra_args])

    assert result.exit_code == 0, repr(result.exception)
    train_mock.assert_called_once()
    config = train_mock.call_args.kwargs["config"]
    assert config.dataset == "some/dataset"
    assert config.peft_advanced_params == expected_peft_advanced_params
