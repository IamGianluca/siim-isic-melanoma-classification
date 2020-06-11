import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from siim_isic_melanoma_classification.constants import data_path, train_fpath


def main():
    """Split train/valid set."""
    train_df = pd.read_csv(train_fpath)

    patient_ids = train_df["patient_id"].unique()
    patient_means = train_df.groupby(["patient_id"])["target"].mean()

    # split at patient_id level
    train_idx, val_idx = train_test_split(
        np.arange(len(patient_ids)),
        stratify=(patient_means > 0),
        test_size=0.2,  # validation set size
    )

    train_patient_ids = patient_ids[train_idx]
    train = train_df[train_df.patient_id.isin(train_patient_ids)]
    print(f"Samples in train set: {train.shape[0]:,}")
    print(f"Pct of pos. in train set: {train.target.mean():.2%}")

    valid_patient_ids = patient_ids[val_idx]
    valid = train_df[train_df.patient_id.isin(valid_patient_ids)]
    print(f"Samples in valid set: {valid.shape[0]:,}")
    print(f"Pct of pos. in valid set: {valid.target.mean():.2%}")

    assert train_df.shape[0] == train.shape[0] + valid.shape[0]

    train.to_csv(data_path / "train_patients.csv")
    valid.to_csv(data_path / "valid_patients.csv")


if __name__ == "__main__":
    main()
