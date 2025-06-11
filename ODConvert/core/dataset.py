from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from enum import Enum

import cv2

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

    def get_shape(self) -> tuple[int, int]:
        img = cv2.imread(str(self.path))
        return img.shape[:2]


@dataclass(frozen=True)
class DatasetAnnotation:
    id: int | None
    cls: DatasetClass
    bbox: BoundingBox
    image: DatasetImage
    iscrowd: int
