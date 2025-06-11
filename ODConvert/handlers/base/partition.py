from typing import List, Dict, Tuple, final, TYPE_CHECKING
from abc import abstractmethod

from ODConvert.core.dataset import DatasetClass, DatasetAnnotation, DatasetImage

if TYPE_CHECKING:
    from .handler import DatasetHandler

class DatasetPartitionHandler:

    """
    DatasetPartitionHandler is a base class for handling dataset partitions.

    This class provides a framework for partitioning a dataset into subsets, such as training, validation, or testing sets.

    :param name: The name of the dataset partition.
    :param parent: The parent dataset handler of type DatasetHandler.
    """

    def __init__(self, name: str, parent: "DatasetHandler"):
        self.name = name
        self.parent = parent
        # Initialize the classes, annotations, and images
        self.__classes: List[DatasetClass] = self.find_classes()
        self.__images: Dict[int, DatasetImage] = self.find_images()
        # Annotations should be loaded last, they rely on classes and images
        self.__annotations: List[DatasetAnnotation] = self.find_annotations()

    @abstractmethod
    def find_classes(self) -> List[DatasetClass]:
        """
        Expose the classes of the dataset partition.
        :return: List[DatasetClass]
        """
        pass

    @final
    def get_classes(self) -> List[DatasetClass]:
        if self.__classes is not None:
            # If classes are already loaded, return them
            return self.__classes
        # Otherwise, throw an exception
        raise ValueError("Classes not loaded yet.")

    @abstractmethod
    def find_annotations(self) -> List[DatasetAnnotation]:
        """
        Expose the annotations of the dataset partition.
        :return: List[DatasetAnnotation]
        """
        pass

    @final
    def get_annotations(self) -> List[DatasetAnnotation]:
        if self.__annotations is not None:
            # If annotations are already loaded, return them
            return self.__annotations
        # Otherwise, throw an exception
        raise ValueError("Annotations not loaded yet.")

    @abstractmethod
    def find_images(self) -> Dict[int, DatasetImage]:
        """
        Expose the images of the dataset partition.
        :return: List[DatasetImage]
        """
        pass

    @final
    def get_images(self) -> Dict[int, DatasetImage]:
        if self.__images is not None:
            # If images are already loaded, return them
            return self.__images
        # Otherwise, throw an exception
        raise ValueError("Images not loaded yet.")

    @final
    def get_image(self, id: int) -> DatasetImage:
        if self.__images is not None:
            # As long as images are loaded
            # return the matching one
            return self.__images[id]
        # Otherwise, throw an exception
        raise ValueError("Images not loaded yet.")

    def stats(self) -> Tuple[int, int]:
        """
        Returns the number of images and annotations in the dataset partition.
        :return: Tuple[int, int]
        """
        images = self.get_images()
        annotations = self.get_annotations()
        return len(images), len(annotations)
