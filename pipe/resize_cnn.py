from functools import partial
from multiprocessing import Pool

import pandas as pd
from PIL import Image
from siim_isic_melanoma_classification.constants import (
    data_path,
    test_fpath,
    train_fpath,
)

sz = 224


def main():
    train_df = pd.read_csv(train_fpath)
    test_df = pd.read_csv(test_fpath)

    for df, name in [(train_df, "train"), (test_df, "test")]:
        print(f"Resizing {name} images to {sz}x{sz}...")
        resize_images(df, name=name)


def resize_images(df, name: str):
    pool = Pool()
    routine = partial(resize_image, name=name)
    pool.map(routine, df["image_name"])


def resize_image(image_id, name: str, sz=sz):
    image = Image.open(data_path / f"{name}/{image_id}.jpg")
    smaller_image = image.resize((sz, sz))
    out_path = data_path / f"{name}_{sz}"
    if not out_path.exists():  # TODO: move out, optimization
        out_path.mkdir()
    smaller_image.save(out_path / f"{image_id}.jpg")


if __name__ == "__main__":
    main()
