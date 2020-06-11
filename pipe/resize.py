from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

from siim_isic_melanoma_classification.constants import (
    data_path,
    test_fpath,
    train_fpath,
)

IMAGE_SZ = 32


def main():
    train_df = pd.read_csv(train_fpath)
    test_df = pd.read_csv(test_fpath)

    for df, name in [(train_df, "train"), (test_df, "test")]:
        print(f"Converting {name} images to NumPy array...")
        ar = convert_to_array(df, name=name)
        np.save(data_path / f"x_{name}_32", ar)


def convert_to_array(df, name: str):
    pool = Pool()
    routine = partial(image_to_flattened_array, name=name)
    data = pool.map(routine, df["image_name"])
    return np.vstack(data)


def image_to_flattened_array(image_id, name: str, desired_size=IMAGE_SZ):
    image_fpath = data_path / f"{name}/{image_id}.jpg"
    im = Image.open(image_fpath)
    small_im = im.resize((desired_size,) * 2, resample=Image.LANCZOS)
    return np.array(small_im).reshape((1, 32, 32, 3))


if __name__ == "__main__":
    main()
