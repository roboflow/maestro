# import typer

# from maestro.cli.introspection import find_training_recipes

# app = typer.Typer()
# find_training_recipes(app=app)


# @app.command(help="Display information about maestro")
# def info():
#     typer.echo("Welcome to maestro CLI. Let's train some VLM! 🏋")


# if __name__ == "__main__":
#     app()

import typer
from maestro.trainer.models.florence_2.entrypoint import florence2_app

app = typer.Typer()
app.add_typer(florence2_app, name="florence2")

if __name__ == "__main__":
    app()
