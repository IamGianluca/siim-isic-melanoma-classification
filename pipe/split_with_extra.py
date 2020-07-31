import numpy as np
import pandas as pd

from siim_isic_melanoma_classification.constants import (
    data_path,
    folds_fpath,
    folds_with_extra_fpath,
)


def main():
    original_folds = pd.read_csv(folds_fpath)
    cols = original_folds.columns

    new_malig = [
        pd.read_csv(data_path / f"train_malig_{n}.csv") for n in range(1, 4)
    ]
    new_malig = pd.concat(new_malig, axis=0)

    # split malignant cases from 2017 and 2018 competition
    # between 5 folds
    condition = [
        new_malig.tfrecord.isin([30, 32, 34]),
        new_malig.tfrecord.isin([36, 38, 40]),
        new_malig.tfrecord.isin([42, 44, 46]),
        new_malig.tfrecord.isin([48, 50, 52]),
        new_malig.tfrecord.isin([54, 56, 58]),
    ]
    choice = [0, 1, 2, 3, 4]
    new_malig["fold"] = np.select(condition, choice, default=-1)

    # remove extra images and cols we don't want yet
    new_malig = new_malig[new_malig.fold != -1]
    new_malig = new_malig.loc[
        :, cols
    ]  # TODO: introduce extra features for meta-model

    folds_with_extra = pd.concat([original_folds, new_malig], axis=0)
    folds_with_extra.to_csv(folds_with_extra_fpath, index=False)


if __name__ == "__main__":
    main()
