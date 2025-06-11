from typing import Dict, List

from ODConvert.core.dataset import DatasetType, DatasetClass
from .partition import DatasetPartitionHandler

class DatasetHandler:

    def __init__(self, typ: DatasetType, classes: List[DatasetClass], partitions: List[DatasetPartitionHandler]):
        # Set the dataset type
        self.__type: DatasetType = typ
        # Convert the provided classes and partitions to dictionaries
        # for faster lookup
        self.__classes: Dict[int, DatasetClass] = {
            cls.id: cls for cls in classes
        }
        self.__partitions: Dict[str, DatasetPartitionHandler] = {
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
        return [x for x in self.__classes.values()]

    def get_partitions(self) -> List[DatasetPartitionHandler]:
        """
        Returns the list of partitions in the dataset.
        :return: List[DatasetPartition]
        """
        return [x for x in self.__partitions.values()]
