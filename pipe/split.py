import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from siim_isic_melanoma_classification.constants import (
    folds_fpath,
    train_fpath,
)


def main():
    df = pd.read_csv(train_fpath)
    # TODO: shall we use a StratifiedGroupKFold strategy also here?
    df_with_folds = assign_fold(df, group_var="patient_id", n_folds=5)
    print(df_with_folds.groupby("fold").target.mean())
    df_with_folds.to_csv(folds_fpath)


def assign_fold(df: pd.DataFrame, group_var: str, n_folds: int):
    df["fold"] = np.empty(df.shape[0])

    kf = GroupKFold(n_splits=n_folds)
    for fold_number, (_, oof_idx) in enumerate(
        kf.split(df, groups=df.loc[:, group_var])
    ):
        df.loc[oof_idx, "fold"] = fold_number

    assert df.fold.isna().sum() == 0
    assert df.fold.min() == 0
    assert df.fold.max() == n_folds - 1
    assert all(df.groupby("patient_id").fold.nunique() == 1)

    return df


if __name__ == "__main__":
    main()
