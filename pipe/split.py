import pandas as pd

from siim_isic_melanoma_classification.constants import (
    dupes_fpath,
    folds_fpath,
    train_fpath,
)
from siim_isic_melanoma_classification.model_selection import (
    assign_fold_using_stratified_group_k_fold,
)


def main():
    # list of duplicate images in training set
    dupes = pd.read_csv(dupes_fpath)
    dupes = dupes[dupes.partition == "train"]
    dupes_image_names = dupes.ISIC_id_paired.tolist()
    dupes_image_names.extend(
        [
            "ISIC_8776686",
            "ISIC_8689583",
            "ISIC_9409255",
            "ISIC_5194874",
            "ISIC_6705662",
            "ISIC_7195645",
            "ISIC_8262759",
            "ISIC_6548307",
            "ISIC_7675261",
        ]
    )

    df = pd.read_csv(train_fpath)

    # remove duplicates
    df = df[~df.image_name.isin(dupes_image_names)].reset_index(drop=True)

    # create folds
    df_with_folds = assign_fold_using_stratified_group_k_fold(
        df, group_var="patient_id", n_folds=5
    )
    print(df_with_folds.groupby("fold").target.mean())
    df_with_folds.to_csv(folds_fpath, index=False)


if __name__ == "__main__":
    main()
