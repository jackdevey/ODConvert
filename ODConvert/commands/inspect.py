from rich import print

from pathlib import Path

import ODConvert.core

from rich.columns import Columns

import fire


def inspect(path: str):
    # Convert the string path to a Path object
    # at the first instance
    path: Path = Path(path)

    if not path.is_dir() or not path.exists():
        # If the path is not a directory, return False
        raise fire.core.FireError(f"Path {path} is not a valid directory")

    dataset = ODConvert.core.autodetect(path)

    dps = dataset.get_partitions()

    classes = dataset.get_classes()

    # Dataset details
    print(f"Path: {path.absolute()}")
    print(f"Type: {dataset.get_type().color_encoded_str()}")
    # Classes
    print()
    print(f"[bold]Detected {len(classes)} classes:[/bold]")
    print(Columns([f"{cls.id:2} → {cls.name}" for cls in classes]))
    # Partitions
    print()
    print(f"[bold]Detected {len(dps)} partitions:[/bold]")
    print(Columns(
        [
            f"[magenta]{dp.name}[/magenta] → {dp.stats()[0]} "
            f"images and {dp.stats()[1]} annotations" for dp in dps
        ],
    ))
