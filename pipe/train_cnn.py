import argparse
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from albumentations.augmentations.transforms import Normalize, Resize
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensor
from pytorch_lightning import Trainer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from siim_isic_melanoma_classification.cnn import MelanomaDataset, MyModel
from siim_isic_melanoma_classification.constants import (
    data_path,
    folds_fpath,
    metrics_path,
    models_path,
    submissions_path,
    test_fpath,
    test_img_224_path,
    train_img_224_path,
)
from siim_isic_melanoma_classification.submit import prepare_submission

arch = "resnet34"
sz = 128
bs = 400
lr = 1e-4
mom = 0.9
epochs = 2

model_fpath = models_path / f"stage1_{arch}.pth"


def main(create_submission: bool = True, crossvalidate: bool = True):
    hparams = dict_to_args(
        {
            "arch": arch,
            "epochs": epochs,
            "sz": sz,
            "bs": bs,
            "lr": lr,
            "mom": mom,
        }
    )

    if crossvalidate:
        # NOTE: we are not going to save the parameters of our DL model here.
        # We are only trying to assess how good the model is
        folds = pd.read_csv(folds_fpath)
        n_folds = folds.fold.nunique()

        oof_predictions = list()
        for fold_number in range(n_folds):
            train_df, valid_df, test_df = train_val_test_split(
                folds, fold_number
            )

            # train model and report valid scores progress during training
            model = MyModel(
                hparams=hparams, train_df=train_df, valid_df=valid_df
            )
            trainer = Trainer(
                gpus=1, max_epochs=hparams.epochs, progress_bar_refresh_rate=0,
            )
            trainer.fit(model)

            # make predictions on OOF data
            transform = Compose(
                [
                    Resize(height=128, width=128),
                    Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    ToTensor(),
                ]
            )
            test_ds = MelanomaDataset(
                df=test_df,
                images_path=train_img_224_path,
                transform=transform,
                train_or_valid=False,
            )
            test_dl = DataLoader(
                test_ds, batch_size=int(bs / 4), shuffle=False, num_workers=8,
            )

            results: List[np.ndarray] = list()
            model.freeze()
            for batch_num, data in enumerate(test_dl):
                data = data.to("cuda")
                results.append(model(data).to("cpu").numpy().squeeze())
            preds = np.concatenate(results, axis=0)

            # store OOF predictions
            preds = pd.DataFrame(
                {"image_name": test_df.image_name, "preds": preds}
            )
            oof_predictions.append(preds)

    oof_preds = pd.concat(oof_predictions).reset_index()

    # sort OOF predictions
    y_true = train_df.loc[:, ["image_name", "target"]]
    r = pd.merge(
        y_true,
        oof_preds,
        how="inner",
        left_on="image_name",
        right_on="image_name",
    )

    # save OOF predictions for L2 meta-model (stacking)
    r.to_csv(data_path / "l1_resnet_oof_predictions.csv", index=False)

    # use OOF predictions to compute CV AUC
    cv_score = roc_auc_score(r["target"], r["preds"])
    print(f"{cv_score:.4f}")
    with open(metrics_path / "l1_resnet_cv.metric", "w") as f:
        f.write(f"OOF CV AUC: {cv_score:.4f}")

    # TODO: train model on entire dataset
    print("Train classifier on entire training dataset...")
    model = MyModel(hparams=hparams, train_df=folds, valid_df=valid_df)
    trainer = Trainer(
        gpus=1,
        max_epochs=hparams.epochs,
        # auto_lr_find="lr",
        progress_bar_refresh_rate=0,
        limit_val_batches=0,
    )
    trainer.fit(model)

    trainer.save_checkpoint(model_fpath)

    if create_submission:
        print("Making predictions on test data...")
        # model.load_from_checkpoint(model_fpath)
        model.freeze()

        y_preds = predict(model)

        prepare_submission(
            y_preds=y_preds.cpu().numpy().squeeze(),
            fname="l1_resnet_predictions.csv",
        )


def train_val_test_split(folds, fold_number):
    train_val = folds[folds.fold != fold_number].reset_index(drop=True)
    test_df = folds[folds.fold == fold_number].reset_index(drop=True)

    # split train/val set
    train_df, valid_df = split_train_test_using_stratigy_group_k_fold(
        train_val
    )
    return train_df, valid_df, test_df


def dict_to_args(d):
    args = argparse.Namespace()

    def dict_to_args_recursive(args, d, prefix=""):
        for k, v in d.items():
            if type(v) == dict:
                dict_to_args_recursive(args, v, prefix=k)
            elif type(v) in [tuple, list]:
                continue
            else:
                if prefix:
                    args.__setattr__(prefix + "_" + k, v)
                else:
                    args.__setattr__(k, v)

    dict_to_args_recursive(args, d)
    return args


def plot_a_batch(train_dl: DataLoader):
    for batch_number, sample_batched in enumerate(train_dl):
        if batch_number == 3:
            plt.figure()
            show_moles(images=sample_batched)
            plt.axis("off")
            plt.ioff()
            plt.show()
            break


def show_moles(images):
    images, _ = images["image"], images["label"]
    grid = utils.make_grid(images)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


def split_train_test_using_stratigy_group_k_fold(df):
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


def predict(model):
    # TODO: move this inside LightningModule
    transform = Compose(
        [
            Resize(height=128, width=128),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensor(),
        ]
    )
    test_df = pd.read_csv(test_fpath)
    test_ds = MelanomaDataset(
        df=test_df,
        images_path=test_img_224_path,
        transform=transform,
        train_or_valid=False,
    )
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=8,)
    results: List[torch.Tensor] = list()
    for batch_num, data in enumerate(test_dl):
        data = data.to("cuda")
        results.append(model(data))
    return torch.cat(results, axis=0)


if __name__ == "__main__":
    main(crossvalidate=True, create_submission=True)
