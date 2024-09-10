import typer
from maestro.trainer.models.florence_2.entrypoint import app as florence2_app

app = typer.Typer()
app.add_typer(florence2_app, name="florence2")
