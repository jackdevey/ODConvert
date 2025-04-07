from ODConvert.core import DatasetHandler, DatasetAnnotation, DatasetType
from typing import Dict, List
from pathlib import Path


def convert_to_yolo(dataset: DatasetHandler, base_path: Path):
    """
    Convert a dataset to YOLO format.
    :param dataset: The dataset to convert.
    :return: None
    """
    # Check if the dataset is already in YOLO format
    if dataset.get_type() == DatasetType.YOLO:
        print("Dataset is already in YOLO format.")
        return

    # Create the images and labels paths
    images_path = base_path.joinpath("images")
    labels_path = base_path.joinpath("labels")

    for partition in dataset.get_partitions():
        # Create the directories for the partition
        partition_images_path = images_path.joinpath(partition.name)
        partition_images_path.mkdir(parents=True, exist_ok=True)
        partition_labels_path = labels_path.joinpath(partition.name)
        partition_labels_path.mkdir(parents=True, exist_ok=True)
        # Get the images and annotations for the partition
        images = partition.get_images()
        annotations = partition.get_annotations()

        images_with_annotations: Dict[int, List[DatasetAnnotation]] = {}

        for image in images.values():
            # Create an empty list for images with annotations
            images_with_annotations[image.id] = []

        for annotation in annotations:
            # Get the image ID from the annotation
            image_id = annotation.image.id
            # Append the annotation to the list of annotations for the image
            images_with_annotations[image_id].append(annotation)

        for image_with_annotations in images_with_annotations:
            # Create the image annotation file
            partition_labels_path.joinpath(
                f"{image_with_annotations}.txt").touch(exist_ok=True)
            # Write the annotations to the file
            with open(partition_labels_path.joinpath(
                    f"{image_with_annotations}.txt"), "w") as f:
                for annotation in images_with_annotations[image_with_annotations]:
                    # Get the class ID and bounding box
                    cls_id = annotation.cls.id
                    bbox = annotation.bbox
                    # Write the annotation to the file
                    f.write(
                        f"{cls_id} {bbox.x_center} {bbox.y_center} {bbox.width} {bbox.height}\n")
