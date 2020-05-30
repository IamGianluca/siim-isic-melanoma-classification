import numpy as np

import pandas as pd
from PIL import Image

from tqdm import tqdm


IMAGE_SZ = 32


def main():
    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")

    # loop through the images from the images ids from the target\id dataset
    # then grab the cooresponding image from disk, pre-process, and store
    # in matrix in memory
    for df, name in [(train_df, "train"), (test_df, "test")]:
        ar = convert_to_array(df, name=name)
        np.save(f"./data/x_{name}_32", ar)


def convert_to_array(df, name: str):
    # get the number of training images from the target\id dataset
    N = df.shape[0]
    # create an empty matrix for storing the images
    x_train = np.empty((N, IMAGE_SZ, IMAGE_SZ, 3), dtype=np.uint8)
    for i, image_id in enumerate(tqdm(df["image_name"])):
        x_train[i, :, :, :] = preprocess_image(f"./data/{name}/{image_id}.jpg")
    return x_train


def preprocess_image(image_path, desired_size=IMAGE_SZ):
    im = Image.open(image_path)
    return im.resize((desired_size,) * 2, resample=Image.LANCZOS)


if __name__ == "__main__":
    main()
