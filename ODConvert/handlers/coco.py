from pathlib import Path
from ODConvert.handlers.base import DatasetPartition, DatasetAnnotation, DatasetClass, DatasetImage, BoundingBox, DatasetHandler
import json
from typing import List, Tuple, Optional


class COCODatasetHandler(DatasetHandler):

    def __init__(self, image_dir: Path):
        # Initialize the dataset partition
        self.image_dir = image_dir
        # Find all partitions in the dataset
        partitions = self.__find_partitions()
        super().__init__(partitions)

    def __get_classes(self):
        # TODO: for now will jsut return the classes from the partition
        # but in the future we will have to have multiple partitions - coco
        # will need to check for the same classes over all partitions
        return self.__partitions[0].get_classes()

    def __find_partitions(self):
        partitions: List[DatasetPartition] = []
        for item in self.image_dir.iterdir():
            if item.is_dir():
                # Treat all subdirectories as partitions
                # and create a DatasetPartition object for each
                partition = COCODatasetPartition(
                    name=item.name,
                    image_dir=item,
                    annotation_file=item / "_annotations.coco.json"
                )
                partitions.append(partition)
        # Return the list of partitions
        return partitions


class COCODatasetPartition(DatasetPartition):

    def __init__(self, name, image_dir: Path, annotation_file: Path):
        self.name = name
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        # Load the annotation file and parse it as JSON
        self.raw = json.loads(open(annotation_file, "r").read())
        # Load classes, images and annotations into memory
        self.__classes = self.get_classes()
        self.__images = self.get_images()
        self.__annotations = self.get_annotations()

    def get_class(self, id: int) -> DatasetClass | None:
        """
        Get a class by its ID.
        :param id: The ID of the class.
        :return: DatasetClass object
        """
        # Load classes if not already loaded
        if self.__classes is None:
            self.__classes = self.get_classes()
        # Search for the class with the given ID
        for cls in self.__classes:
            if cls.id == id:
                return cls
        # If not found, return None
        return None

    def get_classes(self):
        # Check if classes are already loaded
        # and return them if so
        if getattr(self, "__classes", None) is not None:
            return self.__classes

        return [
            # Construct DatasetClass object
            DatasetClass(
                id=category["id"],
                name=category["name"],
                parent=None
            )
            # for all categories in the raw data
            for category in self.raw["categories"]]

    def get_images(self):
        # Check if images are already loaded,
        # and return them if so
        if getattr(self, "__images", None) is not None:
            return self.__images

        return [
            DatasetImage(
                id=image["id"],
                path=self.image_dir / image["file_name"],
            )
            for image in self.raw["images"]
        ]

    def get_annotations(self):
        # Check if annotations are already loaded,
        # and return them if so
        if getattr(self, "__annotations", None) is not None:
            return self.__annotations

        def construct_annotation(annotation):
            # Lookup class by ID
            cls = self.get_class(annotation["category_id"])
            if cls is None:
                raise ValueError(
                    f"Class with ID {annotation['category_id']} not found.")
            # Construct BoundingBox object
            bbox = BoundingBox.from_center(
                annotation["bbox"][0], annotation["bbox"][1], annotation["bbox"][2], annotation["bbox"][3])
            # Construct DatasetAnnotation object
            return DatasetAnnotation(
                id=annotation["id"],
                cls=cls,
                bbox=bbox,
                iscrowd=0
            )

        return [
            # Construct DatasetAnnotation object
            construct_annotation(annotation)
            # for all annotations in the raw data
            for annotation in self.raw["annotations"]
        ]


if __name__ == "__main__":
    # Example usage
    partition = COCODatasetPartition("train", Path(
        "/Users/jack/Developer/ODConvert/.demo/train/"), Path("/Users/jack/Developer/ODConvert/.demo/train/_annotations.coco.json"))
    print(partition.get_classes())  # Output: train
    print(partition.image_dir)  # Output: /path/to/images
    print(partition.get_images())  # Output: /path/to/images/image1.jpg
