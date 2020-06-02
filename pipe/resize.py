from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_SZ = 32


def main():
    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")

    # loop through the images from the images ids from the target\id dataset
    # then grab the cooresponding image from disk, pre-process, and store
    # in matrix in memory
    for df, name in [(train_df, "train"), (test_df, "test")]:
        print(f"Converting {name} images to NumPy array...")
        ar = convert_to_array(df, name=name)
        np.save(f"./data/x_{name}_32", ar)


def convert_to_array(df, name: str):
    pool = Pool()
    routine = partial(image_to_flattened_array, name=name)
    data = pool.map(routine, df["image_name"])
    return np.vstack(data)


def image_to_flattened_array(image_id, name: str, desired_size=IMAGE_SZ):
    image_fpath = f"./data/{name}/{image_id}.jpg"
    im = Image.open(image_fpath)
    small_im = im.resize((desired_size,) * 2, resample=Image.LANCZOS)
    return np.array(small_im).reshape((1, 32, 32, 3))


if __name__ == "__main__":
    main()
