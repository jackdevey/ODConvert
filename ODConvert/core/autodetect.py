from ODConvert.handlers.base import DatasetHandler
from pathlib import Path


def autodetect(path: Path) -> DatasetHandler:
    # Reject fake paths or non-dirs
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"Specified path: {path} does not exist or is not a directory.")

    items = [item for item in path.iterdir()]

    # Check for a COCO dataset by looking for a directory named "annotations"
    # that contains any .json files
    for item in path.iterdir():
        if item.is_dir() and item.name == "annotations":
            # Check if there are any .json files in the annotations directory
            if any(file.suffix == ".json" for file in item.iterdir()):
                # Assume this is a COCO dataset
                from ODConvert.handlers.coco import COCODatasetHandler
                return COCODatasetHandler(path)

    raise TypeError(
        "Unable to detect dataset type. Please specify the dataset type manually."
    )
