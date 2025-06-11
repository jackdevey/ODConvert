from pathlib import Path
from ODConvert.core import DatasetAnnotation, DatasetClass
from ODConvert.core import DatasetImage, BoundingBox
from ODConvert.core import DatasetType
import json
from typing import List
from uuid import uuid4

from ODConvert.handlers.base import DatasetHandler, DatasetPartitionHandler


class COCODatasetHandler(DatasetHandler):

    def __init__(self, dir: Path):
        # Initialise the dataset partition
        self.dir = dir
        # Find all partitions in the dataset
        partitions = self.__find_partitions()
        # Check the first partition for classes
        if not partitions:
            raise ValueError("No partitions found in the dataset.")
        # Check if the first partition has classes
        classes = partitions[0].get_classes()
        if classes is None:
            raise ValueError("No classes found in the dataset.")
        # Initialise the DatasetHandler with the classes and partitions
        # from the first partition
        super().__init__(DatasetType.COCO, classes, partitions)

    def __find_partitions(self):
        partitions: List[DatasetPartitionHandler] = []
        # COCO datasets should provide annotations for each
        # partition in the /annotations directory
        if (self.dir / "annotations").exists():
            for item in (self.dir / "annotations").iterdir():
                # Check if the item is a file and ends with .json
                if item.is_file() and item.suffix == ".json":
                    # Try to extract the partition name from the file name,
                    # typically this comes after the last underscore
                    name = item.stem.split(
                        "_")[-1] or f"unknown-{str(uuid4())[:8]}"
                    # Create new COCODatasetPartitionHandler object
                    # and append it to the list of partitions
                    partitions.append(COCODatasetPartitionHandler(
                        name=name,
                        image_dir=self.dir / "images" / name,
                        annotation_file=item,
                        parent=self
                    ))
        # TODO: Add support for occurences where annotations
        # are stored with images in the partition directories.
        return partitions

        # # Iterate through all items in the annotation directory
        # # and check if they are directories

        # for item in self.image_dir.iterdir():
        #     if item.is_dir():
        #         # Treat all subdirectories as partitions
        #         # and create a DatasetPartition object for each
        #         partition = COCODatasetPartition(
        #             name=item.name,
        #             image_dir=item,
        #             annotation_file=item / "_annotations.coco.json"
        #         )
        #         partitions.append(partition)
        # # Return the list of partitions
        # return partitions


class COCODatasetPartitionHandler(DatasetPartitionHandler):

    def __init__(self, name, image_dir: Path, annotation_file: Path, parent: DatasetHandler):
        self.name = name
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        # Load the annotation file and parse it as JSON
        self.raw = json.loads(open(annotation_file, "r").read())
        # Initialise the base DatasetPartition class
        super().__init__(name, parent)

    def find_classes(self):
        return [
            # Construct DatasetClass object
            DatasetClass(
                id=category["id"],
                name=category["name"],
                parent=None
            )
            # for all categories in the raw data
            for category in self.raw["categories"]
        ]

    def find_images(self):
        return {
            image["id"]: DatasetImage(
                id=image["id"],
                path=self.image_dir / image["file_name"],
            )
            for image in self.raw["images"]
        }

    def get_class(self, id: int) -> DatasetClass | None:
        """
        Get a class by its ID.
        :param id: The ID of the class.
        :return: DatasetClass object
        """
        # Search for the class with the given ID
        for cls in self.get_classes():
            if cls.id == id:
                return cls
        # If not found, return None
        return None

    def find_annotations(self):

        def construct_annotation(annotation):
            # Lookup class by ID
            cls = self.get_class(annotation["category_id"])
            if cls is None:
                raise ValueError(
                    f"Class with ID {annotation['category_id']} not found.")
            # Lookup image by ID
            img = self.get_image(annotation["image_id"])
            if img is None:
                raise ValueError(
                    f"Image with ID {annotation['image_id']} not found."
                )
            # Construct BoundingBox object
            bbox = BoundingBox.from_center(
                annotation["bbox"][0],
                annotation["bbox"][1],
                annotation["bbox"][2],
                annotation["bbox"][3]
            )
            # Construct DatasetAnnotation object
            return DatasetAnnotation(
                id=annotation["id"],
                cls=cls,
                bbox=bbox,
                image=img,
                iscrowd=0
            )

        return [
            # Construct DatasetAnnotation object
            construct_annotation(annotation)
            # for all annotations in the raw data
            for annotation in self.raw["annotations"]
        ]
