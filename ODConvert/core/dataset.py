from dataclasses import dataclass
from abc import abstractmethod
from typing import List, Optional, Tuple, Dict
from pathlib import Path

from ODConvert.core import BoundingBox


@dataclass(frozen=True)
class DatasetClass:
    id: int
    name: str
    parent: Optional["DatasetClass"] = None


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

    def __init__(self, classes: List[DatasetClass], partitions: List[DatasetPartition]):
        # Convert the provided classes and partitions to dictionaries
        # for faster lookup
        self.__classes: Dict[int, DatasetClass] = {
            cls.id: cls for cls in classes
        }
        self.__partitions: Dict[str, DatasetPartition] = {
            partition.name: partition for partition in partitions
        }

    def get_classes(self) -> List[DatasetClass]:
        """
        Returns the list of classes in the dataset.
        :return: List[DatasetClass]
        """
        return self.__classes.values()

    def get_partitions(self) -> List[DatasetPartition]:
        """
        Returns the list of partitions in the dataset.
        :return: List[DatasetPartition]
        """
        return self.__partitions.values()
