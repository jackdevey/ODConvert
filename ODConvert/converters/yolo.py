from ODConvert.core import DatasetAnnotation
from typing import Dict, List
import shutil
from rich.progress import track

from ODConvert.converters.base import DatasetConverter


class YOLOConverter(DatasetConverter):

    def setup(self):
        # Create the images and labels paths
        self.images_path = self.path.joinpath("images")
        self.labels_path = self.path.joinpath("labels")

    def additional_checks(self):
        return True

    def convert_partition(self, partition):
        # Create the directories for the partition
        partition_images_path = self.images_path.joinpath(partition.name)
        partition_images_path.mkdir(parents=True, exist_ok=True)
        partition_labels_path = self.labels_path.joinpath(partition.name)
        partition_labels_path.mkdir(parents=True, exist_ok=True)
        # Get the images and annotations for the partition
        images = partition.get_images()
        annotations = partition.get_annotations()

        # Setup & fill images with annotations dictionary
        images_with_annotations: Dict[int, List[DatasetAnnotation]] = {}
        for image in images.values():
            # Create an empty list for images with annotations
            images_with_annotations[image.id] = []
        for annotation in annotations:
            # Get the image ID from the annotation
            image_id = annotation.image.id
            # Append the annotation to the list of annotations for the image
            images_with_annotations[image_id].append(annotation)

        # Copy images and write labels
        for image_with_annotations in track(
            images_with_annotations,
            description="[white]Copying images and writing labels[/white]"
        ):
            # Copy the image to the partition images path
            image = images[image_with_annotations]
            # Get the height & width of the image
            height, width = image.get_shape()
            # Construct a new file name using the image ID and the original
            # file extension
            new_file_name = f"{image.id}{image.path.suffix}"
            new_file_path = partition_images_path.joinpath(new_file_name)
            # Copy the image to the new file path
            shutil.copyfile(image.path, new_file_path)
            # Create the image annotation file
            partition_labels_path.joinpath(
                f"{image_with_annotations}.txt").touch(exist_ok=True)
            # Write the annotations to the file
            with open(partition_labels_path.joinpath(
                    f"{image_with_annotations}.txt"), "w") as f:
                for annot in images_with_annotations[image_with_annotations]:
                    # Get the class ID and bounding box
                    cls_id = annot.cls.id
                    bbox = annot.bbox
                    # Write the annotation to the file
                    f.write(
                        f"{cls_id} {bbox.x_center / width} "
                        f"{bbox.y_center / height} "
                        f"{bbox.width / width} "
                        f"{bbox.height / height}\n")

    def final_steps(self) -> None:
        # Get all classes from the dataset
        classes = self.dataset.get_classes()
        # Get all partitions from the dataset
        partitions = self.dataset.get_partitions()
        # Create the data.yaml file
        with open(self.path.joinpath("data.yaml"), "w") as f:
            # Write the paths to the dataset
            f.write(f"path: {self.path.absolute()}\n")
            # Write paths for each partition
            for partition in partitions:
                f.write(f"{partition.name}: images/{partition.name}\n")
            f.write("\n")
            # Write the number of classes
            f.write(f"nc: {len(classes)}\n\n")
            # Write the class names
            f.write("names:\n")
            for cls in classes:
                f.write(f"  - {cls.name}\n")
