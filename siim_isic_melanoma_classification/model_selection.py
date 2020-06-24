import random
from collections import Counter, defaultdict
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split


def assign_fold_using_group_k_fold(
    df: pd.DataFrame, group_var: str, n_folds: int, strategy: str
):
    df["fold"] = np.empty(df.shape[0])

    kf = GroupKFold(n_splits=n_folds)
    for fold_number, (_, oof_idx) in enumerate(
        kf.split(df, groups=df.loc[:, group_var])
    ):
        df.loc[oof_idx, "fold"] = fold_number

    assert df.fold.isna().sum() == 0
    assert df.fold.min() == 0
    assert df.fold.max() == n_folds - 1
    assert all(df.groupby(group_var).fold.nunique() == 1)

    return df


def assign_fold_using_stratified_group_k_fold(
    df: pd.DataFrame, group_var: str, n_folds: int
):
    df["fold"] = np.empty(df.shape[0])
    X = df.drop("target", axis=1)
    y = df.loc[:, "target"]
    groups = df.loc[:, group_var]

    kf = StratifiedGroupKFold(n_splits=n_folds)
    for fold_number, (_, oof_idx) in enumerate(
        kf.split(X=X, y=y, groups=groups)
    ):
        df.loc[oof_idx, "fold"] = fold_number

    assert df.fold.isna().sum() == 0
    assert df.fold.min() == 0
    assert df.fold.max() == n_folds - 1
    assert all(df.groupby(group_var).fold.nunique() == 1)

    return df


class StratifiedGroupKFold:
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray], groups
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        yield from stratified_group_k_fold(
            X=X, y=y, groups=groups, k=self.n_splits
        )


def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std(
                [
                    y_counts_per_fold[i][label] / y_distr[label]
                    for i in range(k)
                ]
            )
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(
        groups_and_y_counts, key=lambda x: -np.std(x[1])
    ):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def split_train_test_using_stratify_k_fold(df):
    """Split train/valid set."""
    train_df = df

    patient_ids = train_df["patient_id"].unique()
    patient_means = train_df.groupby(["patient_id"])["target"].mean()

    # split at patient_id level
    train_idx, val_idx = train_test_split(
        np.arange(len(patient_ids)),
        stratify=(patient_means > 0),
        test_size=0.2,  # validation set size
    )

    train_patient_ids = patient_ids[train_idx]
    train = train_df[train_df.patient_id.isin(train_patient_ids)].reset_index()

    valid_patient_ids = patient_ids[val_idx]
    valid = train_df[train_df.patient_id.isin(valid_patient_ids)].reset_index()

    assert train_df.shape[0] == train.shape[0] + valid.shape[0]

    return train, valid
