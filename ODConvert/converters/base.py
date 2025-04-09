from abc import ABC, abstractmethod
from ODConvert.core import DatasetHandler, DatasetType, DatasetPartition
from typing import final
from pathlib import Path
from rich import print


class DatasetConverter(ABC):

    def __init__(self, dataset: DatasetHandler, to: DatasetType, path: Path):
        """
        Initialize the DatasetConverter.
        :param dataset: The dataset to convert.
        :param to: The target dataset type.
        """
        self.dataset = dataset
        self.to = to
        self.path = path
        # Call the setup method to perform any necessary setup
        self.setup()

    @abstractmethod
    def setup(self):
        pass

    @final
    def perform_checks(self):
        # Check if the dataset is already in target format
        if self.dataset.get_type() == self.to:
            raise ValueError("Dataset is already in the target format.")
        # Perform additional custom checks
        if not self.additional_checks():
            # Here just in case the additional checks returns false,
            # ideally it would have already raised an exception
            raise ValueError("Dataset failed additional checks.")

    @abstractmethod
    def additional_checks(self) -> bool:
        pass

    @final
    def convert(self):
        for partition in self.dataset.get_partitions():
            # Print the partition details
            print(
                "[bold]Converting "
                f"[dodger_blue1]{partition.name}[/dodger_blue1] "
                f"partition into {DatasetType.YOLO.color_encoded_str()} "
                "format "
                "[/bold]")
            # Convert each partition
            self.convert_partition(partition)

    @abstractmethod
    def convert_partition(self, partition: DatasetPartition):
        pass
