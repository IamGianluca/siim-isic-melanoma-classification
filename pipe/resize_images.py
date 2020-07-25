from functools import partial
from multiprocessing import Pool

import pandas as pd
from PIL import Image

from siim_isic_melanoma_classification.constants import (
    data_path,
    test_fpath,
    train_fpath,
)

image_sizes = [128, 192, 256, 384, 512, 768, 1024]


def main():
    train_df = pd.read_csv(train_fpath)
    test_df = pd.read_csv(test_fpath)

    for sz in image_sizes:
        for df, name in [(train_df, "train"), (test_df, "test")]:
            # create directory if not exists
            out_path = data_path / f"{name}_{sz}"
            if not out_path.exists():
                out_path.mkdir()

            print(f"Resizing {name} images to {sz}x{sz}...")
            resize_images(df, name=name, sz=sz)


def resize_images(df: pd.DataFrame, name: str, sz: int):
    pool = Pool()
    routine = partial(resize_image, name=name, sz=sz)
    pool.map(routine, df["image_name"])


def resize_image(image_id: str, name: str, sz: int):
    img = Image.open(data_path / f"{name}/{image_id}.jpg")
    try:
        resized_img = resize(img=img, sz=sz, image_id=image_id)
    except ValueError as err:
        print(image_id, img.size, err)
    resized_img.save(data_path / f"{name}_{sz}/{image_id}.jpg")


def resize(img: Image, sz: int, image_id: str):
    if img.size[0] > sz or img.size[1] > sz:
        resized_img = img.resize((sz, sz))
    else:
        resized_img = Image.new("RGB", (sz, sz))
        resized_img.paste(img, (0, 0))
    return resized_img


if __name__ == "__main__":
    main()
