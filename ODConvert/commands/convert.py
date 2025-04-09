from rich import print
from rich.prompt import Confirm

from pathlib import Path

import shutil
import fire
import ODConvert.core

from ODConvert.converters import YOLOConverter


def convert(path: str, to_type: str):
    # Convert the string path to a Path object
    # at the first instance
    path: Path = Path(path)

    # Convert the string to_type to DatasetType
    try:
        to_type: ODConvert.core.DatasetType = ODConvert.core.DatasetType(
            to_type.upper())
    except ValueError:
        # If the to_type is not a valid DatasetType, raise an error
        raise fire.core.FireError(
            f"Invalid dataset type: {to_type}. Valid types are: "
            f"{', '.join([t.value for t in ODConvert.core.DatasetType])}")

    if not path.is_dir() or not path.exists():
        # If the path is not a directory, return False
        raise fire.core.FireError(f"Path {path} is not a valid directory")

    # Load dataset
    dataset = ODConvert.core.autodetect(path)

    # Dataset details
    print("[bold]Existing Dataset Details:[/bold]")
    print(f"Path: {path.absolute()}")
    print(f"Type: {dataset.get_type().color_encoded_str()}")

    # Print plan
    print()
    print("[bold]Output Dataset Details:[/bold]")
    print(f"Path: {path.absolute()}_{to_type.value.lower()}")
    print(f"Type: {to_type.color_encoded_str()}")

    # Ask for confirmation
    print()
    if not Confirm.ask("Confirm produce output dataset?", default=False):
        print("[red]Operation cancelled by user.[/red]")
        raise fire.core.FireExit(0, "Operation cancelled by user.")

    # Try to create the output directory
    # and throw an error if it already exists
    try:
        output_dir = Path(f"{path.absolute()}_{to_type.value.lower()}")
        output_dir.mkdir(exist_ok=False)
    except FileExistsError:
        print(
            f":warning: The planned output directory of "
            f"{path.absolute()}_{to_type.value.lower()} already exists.")
        overide = Confirm.ask(
            "Do you want to override the existing directory?", default=False)
        if overide:
            print(
                f"Overriding existing directory "
                f"{path.absolute()}_{to_type.value.lower()}")
            # Remove the existing directory
            shutil.rmtree(output_dir)
            output_dir.mkdir()
        else:
            raise fire.core.Fire(
                f"Output directory {output_dir} already exists. ")

    print()  # Spacing

    YOLOConverter(dataset, to_type, output_dir).convert()
    print()  # Spacing
    print("[green bold]:white_heavy_check_mark: "
          "Conversion completed successfully![/green bold]")
