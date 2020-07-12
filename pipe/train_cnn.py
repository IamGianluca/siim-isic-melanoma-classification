from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from albumentations.augmentations.transforms import Normalize, Resize
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer
from pytorch_lightning.core.saving import load_hparams_from_yaml
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from siim_isic_melanoma_classification.cnn import MelanomaDataset, MyModel
from siim_isic_melanoma_classification.constants import (
    data_path,
    folds_fpath,
    metrics_path,
    models_path,
    params_fpath,
    test_fpath,
    test_img_224_path,
    train_img_224_path,
)
from siim_isic_melanoma_classification.submit import prepare_submission
from siim_isic_melanoma_classification.utils import dict_to_args
from pytorch_lightning.callbacks import ModelCheckpoint

params = load_hparams_from_yaml(params_fpath)
hparams = dict_to_args(params["train"])

model_fpath = models_path / f"stage1_{hparams.arch}.pth"
logger = MLFlowLogger("logs/")


def main(create_submission: bool = True):
    folds = pd.read_csv(folds_fpath)
    n_folds = folds.fold.nunique()

    oof_preds = list()
    for fold_number in range(n_folds):
        train_df, test_df = split_train_test_sets(folds, fold_number)

        # train model and report valid scores progress during training
        checkpoint_callback = ModelCheckpoint(
            filepath=models_path, monitor="val_loss", mode="min"
        )
        model = MyModel(hparams=hparams, train_df=train_df, valid_df=test_df)
        trainer = Trainer(
            gpus=1,
            max_epochs=hparams.epochs,
            auto_lr_find=True,
            progress_bar_refresh_rate=0,
            # overfit_batches=5,
            logger=logger,
            checkpoint_callback=checkpoint_callback,
        )
        trainer.fit(model)

        # make predictions on OOF data
        # TODO: move into MyModel to avoid duplication and possible synching
        # issues
        augmentations = Compose(
            [
                Resize(height=hparams.sz, width=hparams.sz),
                Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ]
        )
        test_ds = MelanomaDataset(
            df=test_df,
            images_path=train_img_224_path,
            augmentations=augmentations,
            train_or_valid=False,
        )
        test_dl = DataLoader(
            test_ds, batch_size=hparams.bs, shuffle=False, num_workers=8,
        )

        results: List[np.ndarray] = list()

        # load best weights for this fold
        model = MyModel.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )

        model.freeze()
        model.to("cuda")

        for batch_num, data in enumerate(test_dl):
            data = data.to("cuda")
            results.append(model(data).cpu().numpy().reshape(-1,))
        preds = np.concatenate(results, axis=0)

        # store OOF predictions
        preds = pd.DataFrame(
            {"image_name": test_df.image_name, "preds": preds}
        )
        oof_preds.append(preds)

    oof_preds = pd.concat(oof_preds).reset_index()

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
    logger.log_metrics({"oof_cv_auc": cv_score})
    print(f"OOF CV AUC:{cv_score:.4f}")
    with open(metrics_path / "l1_resnet_cv.metric", "w") as f:
        f.write(f"OOF CV AUC: {cv_score:.4f}")

    # # train model on entire dataset
    # print("Train classifier on entire training dataset...")
    # model = MyModel(hparams=hparams, train_df=folds, valid_df=valid_df)
    # trainer = Trainer(
    #     gpus=1,
    #     max_epochs=hparams.epochs,
    #     # auto_lr_find="lr",
    #     progress_bar_refresh_rate=0,
    #     limit_val_batches=0,
    # )
    # trainer.fit(model)

    # trainer.save_checkpoint(model_fpath)

    # if create_submission:
    #     print("Making predictions on test data...")
    #     # model.load_from_checkpoint(model_fpath)
    #     model.freeze()

    #     y_preds = predict(model)

    #     prepare_submission(
    #         y_preds=y_preds.cpu().numpy().squeeze(),
    #         fname="l1_resnet_predictions.csv",
    #     )


def split_train_test_sets(folds, fold_number):
    train_df = folds[folds.fold != fold_number].reset_index(drop=True)
    test_df = folds[folds.fold == fold_number].reset_index(drop=True)
    return train_df, test_df


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
    images, _ = images[0], images[1]
    grid = utils.make_grid(images)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


def split_train_test_using_stratigy_group_k_fold(df):
    """Split train/valid set."""
    train_df = df

    patient_ids = train_df["patient_id"].unique()
    patient_means = train_df.groupby(["patient_id"])["target"].mean()

    # split at patient_id level
    train_idx, val_idx = split_train_test_sets(
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
    augmentations = Compose(
        [
            Resize(height=hparams.sz, width=hparams.sz),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    test_df = pd.read_csv(test_fpath)
    test_ds = MelanomaDataset(
        df=test_df,
        images_path=test_img_224_path,
        augmentations=augmentations,
        train_or_valid=False,
    )
    test_dl = DataLoader(
        test_ds, batch_size=hparams.bs, shuffle=False, num_workers=8,
    )
    results: List[torch.Tensor] = list()
    for batch_num, data in enumerate(test_dl):
        data = data.to("cuda")
        results.append(model(data))
    return torch.cat(results, axis=0)


if __name__ == "__main__":
    main(create_submission=True)
