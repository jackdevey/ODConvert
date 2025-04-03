class BoundingBox:

    def __init__(self):
        # Reject direct instantiation of BoundingBox
        raise RuntimeError(
            "Use BoundingBox.from_center() / BoundingBox.from_min_max() to create an instance."
        )

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
