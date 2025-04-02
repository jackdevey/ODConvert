from rich import print

from pathlib import Path

from utils.detect_type import detect_type

import fire


def inspect(path: str):
    # Convert the string path to a Path object
    # at the first instance
    path: Path = Path(path)

    print("Inspecting path...")

    if not path.is_dir() or not path.exists():
        # If the path is not a directory, return False
        raise fire.core.FireError(f"Path {path} is not a valid directory")

    # Calculate the dataset path
    ds_type = detect_type(path)

    print("[bold magenta]ODConvert[/bold magenta]!")
    print(f"Dataset type: {ds_type}")

    print(f"hello {path.absolute()}")


if __name__ == "__main__":
    inspect(path=".demo")
