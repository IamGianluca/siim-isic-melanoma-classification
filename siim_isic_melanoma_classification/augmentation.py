import os
import random
from pathlib import Path

import cv2
import numpy as np
from albumentations import ImageOnlyTransform

from siim_isic_melanoma_classification.constants import data_path


class AdvancedHairAugmentation(ImageOnlyTransform):
    def __init__(
        self,
        hairs: int = 5,
        hairs_folder: Path = data_path / "hairs",
        always_apply=False,
        p=0.5,
    ):
        self.hairs = hairs
        self.hairs_folder = hairs_folder
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        n_hairs = random.randint(0, self.hairs)

        if not n_hairs:
            return img

        height, width, _ = img.shape  # target image width and height
        hair_images = [
            im for im in os.listdir(self.hairs_folder) if "png" in im
        ]

        for _ in range(n_hairs):
            hair = cv2.imread(
                os.path.join(self.hairs_folder, random.choice(hair_images))
            )
            hair = cv2.cvtColor(hair, cv2.COLOR_BGR2RGB)
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho : roi_ho + h_height, roi_wo : roi_wo + h_width]

            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            dst = cv2.add(img_bg, hair_fg, dtype=cv2.CV_64F)
            img[roi_ho : roi_ho + h_height, roi_wo : roi_wo + h_width] = dst

        return img


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
