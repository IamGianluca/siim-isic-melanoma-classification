import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from constants import (
    data_path,
    test_array_image_fpath,
    test_fpath,
    train_array_image_fpath,
    train_fpath,
)

IMAGE_SZ = 32


def main():
    train_df = pd.read_csv(train_fpath)
    test_df = pd.read_csv(test_fpath)

    # loop through the images from the images ids from the target id dataset
    # then grab the cooresponding image from disk, pre-process, and store
    # in matrix in memory
    for df, name, fpath in [
        (train_df, "train", train_array_image_fpath),
        (test_df, "test", test_array_image_fpath),
    ]:
        ar = convert_to_array(df, name=name)
        np.save(fpath, ar)


def convert_to_array(df, name: str):
    # get the number of training images from the target\id dataset
    N = df.shape[0]
    # create an empty matrix for storing the images
    x_train = np.empty((N, IMAGE_SZ, IMAGE_SZ, 3), dtype=np.uint8)
    for i, image_id in enumerate(tqdm(df["image_name"])):
        x_train[i, :, :, :] = preprocess_image(
            data_path / f"{name}/{image_id}.jpg"
        )
    return x_train


def preprocess_image(image_path, desired_size=IMAGE_SZ):
    im = Image.open(image_path)
    return im.resize((desired_size,) * 2, resample=Image.LANCZOS)


if __name__ == "__main__":
    main()
