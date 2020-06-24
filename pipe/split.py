import pandas as pd

from siim_isic_melanoma_classification.constants import (
    folds_fpath,
    train_fpath,
)
from siim_isic_melanoma_classification.model_selection import (
    assign_fold_using_stratified_group_k_fold,
)


def main():
    df = pd.read_csv(train_fpath)
    df_with_folds = assign_fold_using_stratified_group_k_fold(
        df, group_var="patient_id", n_folds=5
    )
    print(df_with_folds.groupby("fold").target.mean())
    df_with_folds.to_csv(folds_fpath)


if __name__ == "__main__":
    main()
