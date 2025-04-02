from pathlib import Path
from dataclasses import dataclass
from abc import abstractmethod
from typing import List, Optional


@dataclass(frozen=True)
class DatasetClass:
    id: int
    name: str
    parent: Optional["DatasetClass"] = None


@dataclass(frozen=True)
class BoundingBox:
    x_center: float
    y_center: float
    width: float
    height: float

    def __init__(self):
        # Reject direct instantiation of BoundingBox
        raise RuntimeError(
            "Use BoundingBox.from_center() / BoundingBox.from_min_max() to create an instance.")

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

    def get_area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class DatasetAnnotation:
    id: str | None
    cls: DatasetClass
    bbox: BoundingBox
    iscrowd: int = 0


@dataclass(frozen=True)
class DatasetImage:
    id: str | None
    path: Path
    annotations: List[DatasetAnnotation]


class DatasetPartition:

    @abstractmethod
    def get_annotations(self) -> List[DatasetAnnotation]:
        pass

    @abstractmethod
    def get_images(self) -> List[DatasetImage]:
        pass


if __name__ == "__main__":
    # BoundingBox.from_center(0.5, 0.5, 1.0, 1.0)
    BoundingBox.from_min_max(0.0, 0.0, 1.0, 1.0)
