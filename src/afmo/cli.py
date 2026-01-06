"""Utilities for AFMo (src/afmo/cli.py)."""

import typer

app = typer.Typer(help="AFMO CLI")

def main():
    app()

if __name__ == "__main__":
    main()