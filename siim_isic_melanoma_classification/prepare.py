import numpy as np
import pandas as pd

from siim_isic_melanoma_classification.constants import (
    N_TEST,
    N_TRAIN,
    test_array_image_fpath,
    test_fpath,
    train_array_image_fpath,
    train_fpath,
)


def prepare_dataset(name: str):
    if name == "train":
        array_image_fpath = train_array_image_fpath
        fpath = train_fpath
        N = N_TRAIN
    elif name == "test":
        array_image_fpath = test_array_image_fpath
        fpath = test_fpath
        N = N_TEST
    else:
        raise ValueError("name should be either `train` or `test`")

    image_ar = np.load(array_image_fpath)
    flat_image_ar = image_ar.reshape(N, 32 * 32 * 3)
    flat_image_df = pd.DataFrame(flat_image_ar)

    context = pd.read_csv(fpath)
    y = context.get("target", None)
    try:
        context_vars = context.drop(
            ["diagnosis", "target", "benign_malignant"], axis=1
        )
    except KeyError:
        context_vars = context

    X = pd.concat([flat_image_df, context_vars], axis=1)
    return X, y
