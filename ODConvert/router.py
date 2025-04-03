from rich import print

from pathlib import Path

from ODConvert.utils.detect_type import detect_type

from ODConvert.handlers.coco import COCODatasetPartition, COCODatasetHandler

from rich.columns import Columns

import fire


def inspect(path: str):
    # Convert the string path to a Path object
    # at the first instance
    path: Path = Path(path)

    print("Inspecting path...")

    if not path.is_dir() or not path.exists():
        # If the path is not a directory, return False
        raise fire.core.FireError(f"Path {path} is not a valid directory")

    dataset = COCODatasetHandler(path)

    dps = dataset.get_partitions()

    classes = dataset.get_classes()

    print(f"full path: {path.absolute()}")
    # Classes
    print()
    print(f"[bold]Detected {len(classes)} classes:[/bold]")
    print(Columns([f"{cls.id:2} → {cls.name}" for cls in classes]))
    # Partitions
    print()
    print(f"[bold]Detected {len(dps)} partitions:[/bold]")
    print(Columns(
        [f"[magenta]{dp.name}[/magenta] → {dp.stats()[0]} images and {dp.stats()[1]} annotations" for dp in dps],
    ))
    # print(f"[bold]Detected {len(images)} images[/bold]")
    # print(f"[bold]Detected {len(annotations)} annotations[/bold]")


if __name__ == "__main__":
    inspect(path=".demo/train")
