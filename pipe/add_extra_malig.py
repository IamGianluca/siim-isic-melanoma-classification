import os
import shutil
from shutil import copytree
from zipfile import ZipFile

from tensorboard.compat import tf

from siim_isic_melanoma_classification.constants import data_path

szs = ["128", "192", "256", "384", "512"]


def main():

    for sz in szs:
        source_path = data_path / f"train_{sz}"
        destination_path = data_path / f"train_with_extra_{sz}"

        if destination_path.exists():
            destination_path.rmdir()
            destination_path.mkdir()

        # copy original train dataset
        print(f"Copying {sz}x{sz} training images...")
        copytree(src=source_path, dst=destination_path)

        # unseen images from ISIC's online gallery
        print(f"Extracting {sz}x{sz} images from ISIC's online gallery...")
        archive = ZipFile(f"./malignant-v2-{sz}x{sz}.zip")
        for f in archive.namelist():
            if f.startswith("jpeg"):
                fname = os.path.basename(f)
                source_to = archive.open(f)

                with open(destination_path / fname, "wb") as target:
                    shutil.copyfileobj(source_to, target)

        # images from 2017, 2018, and 2019 competitions
        print(f"Extracting {sz}x{sz} images from 2019 challenge...")
        archive = ZipFile(f"./jpeg-isic2019-{sz}x{sz}.zip")
        for f in archive.namelist():
            if f.endswith(".jpg"):
                fname = os.path.basename(f)
                source_to = archive.open(f)

                with open(destination_path / fname, "wb") as target:
                    shutil.copyfileobj(source_to, target)

    # extract meta-data for these extra images
    archive = ZipFile("./malignant-v2-128x128.zip")
    for f in archive.namelist():
        if f.endswith(".csv"):
            archive.extract(f, data_path)


if __name__ == "__main__":
    main()
