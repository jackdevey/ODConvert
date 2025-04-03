from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class DatasetClass:
    id: int
    name: str
    parent: Optional["DatasetClass"] = None


class BoundingBox:

    def __init__(self):
        # Reject direct instantiation of BoundingBox
        raise RuntimeError(
            "Use BoundingBox.from_center() / BoundingBox.from_min_max() to create an instance."
        )

    @classmethod
    def from_center(cls, x_center: float, y_center: float, width: float, height: float) -> "BoundingBox":
        self = object.__new__(cls)
        object.__setattr__(self, 'x_center', x_center)
        object.__setattr__(self, 'y_center', y_center)
        object.__setattr__(self, 'width', width)
        object.__setattr__(self, 'height', height)
        return self

    @classmethod
    def from_min_max(cls, min_x: float, min_y: float, max_x: float, max_y: float) -> "BoundingBox":
        self = object.__new__(cls)
        object.__setattr__(self, 'width', max_x - min_x)
        object.__setattr__(self, 'height', max_y - min_y)
        object.__setattr__(self, 'x_center', min_x + self.width / 2)
        object.__setattr__(self, 'y_center', min_y + self.height / 2)
        return self


@dataclass(frozen=True)
class DatasetAnnotation:
    id: int | None
    cls: DatasetClass
    bbox: BoundingBox
    iscrowd: int


@dataclass(frozen=True)
class DatasetImage:
    id: int | None
    path: Path


class DatasetPartition:
    image_dir: Path
    annotation_file: Path

    @abstractmethod
    def get_classes(self) -> List[DatasetClass]:
        pass

    @abstractmethod
    def get_annotations(self) -> List[DatasetAnnotation]:
        pass

    @abstractmethod
    def get_images(self) -> List[DatasetImage]:
        pass

    def stats(self) -> Tuple[int, int]:
        """
        Returns the number of images and annotations in the dataset partition.
        :return: Tuple[int, int]
        """
        images = self.get_images()
        annotations = self.get_annotations()
        return len(images), len(annotations)


class DatasetHandler:

    def __init__(self, partitions: List[DatasetPartition]):
        # Find all partitions in the dataset
        # and store them in the instance
        self.__partitions = partitions

    def get_classes(self) -> List[DatasetClass]:
        """
        Returns the list of classes in the dataset.
        :return: List[DatasetClass]
        """
        # Check if classes are already loaded
        # and return them if so
        if getattr(self, "__classes", None) is not None:
            return self.__classes
        # Otherwise, load the classes
        # and store them in the instance
        self.__classes = self.__get_classes()
        return self.__classes

    @abstractmethod
    def __get_classes(self) -> List[DatasetClass]:
        pass

    def get_partitions(self) -> List[DatasetPartition]:
        """
        Returns the list of partitions in the dataset.
        :return: List[DatasetPartition]
        """
        # Partitions are already loaded in the constructor
        # so we just return them
        return self.__partitions
