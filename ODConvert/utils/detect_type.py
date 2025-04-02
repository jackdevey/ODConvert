from pathlib import Path


def detect_type(path: Path) -> str:

    def check_is_yolo(path: Path) -> bool:
        # TODO
        return True

    if check_is_yolo(path):
        return "yolo"
