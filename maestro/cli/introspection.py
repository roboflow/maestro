import os

import typer

from maestro.cli.env import DEFAULT_DISABLE_RECIPE_IMPORTS_WARNINGS_ENV, DISABLE_RECIPE_IMPORTS_WARNINGS_ENV
from maestro.cli.utils import str2bool


def find_training_recipes(app: typer.Typer) -> None:
    try:
        from maestro.trainer.models.florence_2.entrypoint import florence_2_app

        app.add_typer(florence_2_app, name="florence_2")
    except Exception:
        _warn_about_recipe_import_error(model_name="Florence-2")

    try:
        from maestro.trainer.models.paligemma_2.entrypoint import paligemma_2_app

        app.add_typer(paligemma_2_app, name="paligemma_2")
    except Exception:
        _warn_about_recipe_import_error(model_name="PaliGemma 2")

    # try:
    #     from maestro.trainer.models.qwen_2_5_vl.entrypoint import qwen_2_5_vl_app
    #
    #     app.add_typer(qwen_2_5_vl_app, name="qwen_2_5_vl")
    # except Exception:
    #     _warn_about_recipe_import_error(model_name="Qwen2.5-VL")

    from maestro.trainer.models.qwen_2_5_vl.entrypoint import qwen_2_5_vl_app

    app.add_typer(qwen_2_5_vl_app, name="qwen_2_5_vl")


def _warn_about_recipe_import_error(model_name: str) -> None:
    disable_warnings = str2bool(
        os.getenv(
            DISABLE_RECIPE_IMPORTS_WARNINGS_ENV,
            DEFAULT_DISABLE_RECIPE_IMPORTS_WARNINGS_ENV,
        )
    )
    if disable_warnings:
        return None
    warning = typer.style("WARNING", fg=typer.colors.RED, bold=True)
    message = "🚧 " + warning + f" cannot import recipe for {model_name}"
    typer.echo(message)
