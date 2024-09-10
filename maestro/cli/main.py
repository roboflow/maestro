import typer

from maestro.cli.introspection import find_training_recipes

app = typer.Typer()
find_training_recipes(app=app)


@app.command(help="Display information about maestro")
def info():
    typer.echo("Welcome to maestro CLI. Let's train some VLM! 🏋")


if __name__ == "__main__":
    app()
