import random

import cv2
import numpy as np


class Microscope:
    """
    Cutting out the edges around the center circle of the image
    Imitating a picture, taken through the microscope

    Args:
        p (float): probability of applying an augmentation
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to apply transformation to.

        Returns:
            PIL Image: Image with transformation.
        """
        if random.random() < self.p:
            circle = cv2.circle(
                (np.ones(img.shape) * 255).astype(
                    np.uint8
                ),  # image placeholder
                (
                    img.shape[0] // 2,
                    img.shape[1] // 2,
                ),  # center point of circle
                random.randint(
                    img.shape[0] // 2 - 3, img.shape[0] // 2 + 15
                ),  # radius
                (0, 0, 0),  # color
                -1,
            )

            mask = circle - 255
            img = np.multiply(img, mask)

        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"
