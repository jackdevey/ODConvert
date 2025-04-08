from dataclasses import dataclass
from abc import abstractmethod
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from enum import Enum

from ODConvert.core import BoundingBox


class DatasetType(Enum):
    YOLO = "YOLO"
    COCO = "COCO"
    VOC = "VOC"

    def __str__(self):
        """
        Returns the string representation of the dataset type.
        :return: str
        """
        return self.value

    def color(self) -> str:
        """
        Returns the frontend color associated with the dataset type.
        :return: str
        """
        match self:
            case DatasetType.YOLO:
                return "green"
            case DatasetType.COCO:
                return "orange3"
            case DatasetType.VOC:
                return "red"

    def color_encoded_str(self) -> str:
        """
        Returns the string representation of the dataset type
        with color encoding.
        :return: str
        """
        return f"[{self.color()}]{self}[/{self.color()}]"


@dataclass(frozen=True)
class DatasetClass:
    id: int
    name: str
    parent: Optional["DatasetClass"] = None


@dataclass(frozen=True)
class DatasetImage:
    id: int | None
    path: Path


@dataclass(frozen=True)
class DatasetAnnotation:
    id: int | None
    cls: DatasetClass
    bbox: BoundingBox
    image: DatasetImage
    iscrowd: int


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
    def get_images(self) -> Dict[int, DatasetImage]:
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

    def __init__(self,
                 typ: DatasetType,
                 classes: List[DatasetClass],
                 partitions: List[DatasetPartition]
                 ):
        # Set the dataset type
        self.__type: DatasetType = typ
        # Convert the provided classes and partitions to dictionaries
        # for faster lookup
        self.__classes: Dict[int, DatasetClass] = {
            cls.id: cls for cls in classes
        }
        self.__partitions: Dict[str, DatasetPartition] = {
            partition.name: partition for partition in partitions
        }

    def get_type(self) -> DatasetType:
        """
        Returns the type of the dataset.
        :return: DatasetType
        """
        return self.__type

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
