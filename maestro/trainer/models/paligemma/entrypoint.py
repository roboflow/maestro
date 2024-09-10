import typer

paligemma_app = typer.Typer(help="Fine-tune and evaluate PaliGemma model")


@paligemma_app.command(help="Train PaliGemma model")
def train() -> None:
    typer.echo("🚧 Just a placeholder - to be implemented 🚧")


@paligemma_app.command(help="Evaluate PaliGemma model")
def evaluate() -> None:
    typer.echo("🚧 Just a placeholder - to be implemented 🚧")
