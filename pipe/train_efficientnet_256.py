import os
from pathlib import Path
from typing import List, Optional

import albumentations.augmentations.transforms as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from albumentations.core.composition import Compose
from albumentations.pytorch.transforms import ToTensorV2
from efficientnet_pytorch.model import EfficientNet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.saving import load_hparams_from_yaml
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.metrics.functional import auroc
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from siim_isic_melanoma_classification.augmentation import (
    AdvancedHairAugmentation,
)
from siim_isic_melanoma_classification.cnn import (
    Flatten,
    FocalLoss,
    MelanomaDataset,
)
from siim_isic_melanoma_classification.constants import (
    data_path,
    folds_fpath,
    folds_with_extra_fpath,
    metrics_path,
    models_path,
    params_fpath,
    submissions_path,
    test_fpath,
    test_img_256_path,
    train_img_256_extra_path,
    train_img_256_path,
)
from siim_isic_melanoma_classification.lr_scheduler import (
    DelayedCosineAnnealingLR,
)
from siim_isic_melanoma_classification.over9000 import Over9000
from siim_isic_melanoma_classification.submit import prepare_submission
from siim_isic_melanoma_classification.utils import dict_to_args

params = load_hparams_from_yaml(params_fpath)
hparams = dict_to_args(params["train_efficientnet_256"])
logger = MLFlowLogger("logs/")

name = "efficientnet"
oof_preds_fpath = data_path / f"l1_{name}_{hparams.sz}_oof_preds.csv"
metric_fpath = metrics_path / f"l1_{name}_{hparams.sz}_cv.metric"
submission_fpath = submissions_path / f"l1_{name}_{hparams.sz}_submission.csv"


def main(create_submission: bool = True):
    folds = pd.read_csv(folds_fpath)
    n_folds = folds.fold.nunique()

    oof_preds = list()
    ckpt_fpaths = list()
    for fold_number in range(n_folds):
        path = (
            data_path / f"oof_preds_{name}_{hparams.sz}_fold{fold_number}.csv"
        )
        ckpt_fpath = train(fold_number=fold_number, folds=folds, path=path)
        ckpt_fpaths.append(ckpt_fpath)
        oof_preds.append(pd.read_csv(path))

    # save oof preds for stacking
    sorted_oof_preds = sort_oof_predictions(oof_preds, folds)
    sorted_oof_preds.to_csv(oof_preds_fpath, index=False)

    # oof cv score
    cv_score = roc_auc_score(
        sorted_oof_preds["target"], sorted_oof_preds["preds"]
    )
    logger.log_metrics({"oof_cv_auc": cv_score})
    print(f"OOF CV AUC:{cv_score:.4f}")
    with open(metric_fpath, "w") as f:
        f.write(f"OOF CV AUC: {cv_score:.4f}")

    # predict on test data
    preds = list()
    for fold_number in range(n_folds):
        ckpt_fpath = ckpt_fpaths[fold_number]
        path = (
            submissions_path
            / f"subs_{name}_{hparams.sz}_fold{fold_number}.csv"
        )
        model_preds = inference(ckpt_fpath=ckpt_fpath, path=path)
        preds.append(model_preds)
    preds_df = pd.concat([p["preds"] for p in preds], axis=1).reset_index(
        drop=True
    )

    # average predictions from 5 different L1 models
    predictions = preds_df.mean(axis=1)
    prepare_submission(y_pred=predictions, fpath=submission_fpath)


def sort_oof_predictions(oof_preds, folds):
    oof_preds = pd.concat(oof_preds).reset_index()
    y_true = folds.loc[:, ["image_name", "target"]]
    results = pd.merge(
        y_true,
        oof_preds,
        how="inner",
        left_on="image_name",
        right_on="image_name",
    )
    return results


def split_train_test_sets(folds, fold_number, use_extra_data=False):
    if use_extra_data:
        folds_with_extra = pd.read_csv(folds_with_extra_fpath)
        train_df = folds_with_extra[
            folds_with_extra.fold != fold_number
        ].reset_index(drop=True)
    else:
        train_df = folds[folds.fold != fold_number].reset_index(drop=True)
    test_df = folds[folds.fold == fold_number].reset_index(drop=True)
    return train_df, test_df


def train(folds: pd.DataFrame, fold_number: int, path):
    """Create OOF predictions for fold_number."""
    train_df, test_df = split_train_test_sets(
        folds, fold_number, use_extra_data=True
    )

    # train model and report valid scores progress during training
    model = MyModel(
        hparams=hparams,
        model_name=name,
        fold=fold_number,
        train_df=train_df,
        valid_df=test_df,
        test_df=test_df,
        train_images_path=train_img_256_extra_path,
        valid_images_path=train_img_256_path,
        test_images_path=train_img_256_path,  # NOTE: OOF predictions
        path=path,
    )
    callback = ModelCheckpoint(
        filepath=models_path / "{model_name}_{sz}_{fold}",
        monitor="val_auc",
        mode="max",
        save_weights_only=True,
    )

    print("O2" if hparams.precision == 32 else "O1")
    trainer = Trainer(
        gpus=1,
        max_epochs=hparams.epochs,
        # overfit_batches=5,
        num_sanity_val_steps=5,
        amp_level="O2" if hparams.precision == 32 else "O1",
        precision=hparams.precision,
        accumulate_grad_batches=16,
        logger=logger,
        checkpoint_callback=callback,
    )
    # find optimal learning rate
    if hparams.lr == "find":
        trainer.scaler = torch.cuda.amp.GradScaler()
        lr_finder = trainer.lr_find(model)
        optimal_lr = lr_finder.suggestion()
        print(f"Optimal LR: {optimal_lr}")
        model.lr = optimal_lr

    trainer.fit(model)

    # make predictions on OOF data
    print(f"Loading {callback.best_model_path}")
    trainer.test(ckpt_path=callback.best_model_path)
    return callback.best_model_path


def inference(
    ckpt_fpath: str, path,
):
    test_df = pd.read_csv(test_fpath)

    # load best weights for this fold
    model = MyModel.load_from_checkpoint(checkpoint_path=ckpt_fpath)
    model.freeze()
    model.to("cuda")

    # make dataloader load full test data
    model.test_df = test_df
    model.test_images_path = test_img_256_path

    results = list()
    for batch in model.test_dataloader():
        batch = batch.to("cuda")
        results.append(model(batch).cpu().numpy().reshape(-1))
    preds = np.concatenate(results, axis=0)
    preds_df = pd.DataFrame({"image_name": test_df.image_name, "preds": preds})
    preds_df.to_csv(path, index=False)
    return preds_df


class MyModel(LightningModule):
    def __init__(
        self,
        hparams,
        model_name: Optional[str] = None,
        fold: Optional[int] = None,
        train_df: Optional[pd.DataFrame] = None,
        valid_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        train_images_path: Optional[Path] = None,
        valid_images_path: Optional[Path] = None,
        test_images_path: Optional[Path] = None,
        path: Optional[Path] = None,
    ):
        super().__init__()
        self.path = path
        self.model_name = model_name
        self.sz = 256
        self.hparams = hparams
        self.lr = self.hparams.lr
        self.fold = fold

        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

        self.train_images_path = train_images_path
        self.valid_images_path = valid_images_path
        self.test_images_path = test_images_path

        if "resnet" in self.hparams.arch:
            self.model = models.__dict__[self.hparams.arch](pretrained=True)
            n_in = self.model.fc.in_features
            remove_range = 2
            self.model = nn.Sequential(
                *list(self.model.children())[:-remove_range]
            )
            n_out = 1
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Dropout(),
                nn.Linear(n_in, n_out),
            )

        # if "resnext" in self.hparams.arch:
        #     self.model = torch.hub.load(
        #         "facebookresearch/semi-supervised-ImageNet1K-models",
        #         self.hparams.arch,
        #     )
        #     c_feature = self.model.fc.in_features
        #     remove_range = 2  # TODO: ditto
        if "efficient" in self.hparams.arch:
            self.model = EfficientNet.from_pretrained(
                self.hparams.arch, advprop=True, num_classes=1
            )
            # self.head = nn.Sequential(
            #     nn.ReLU(), nn.Dropout(), nn.Linear(1000, 1)
            # )

    def train_dataloader(self):
        augmentations = Compose(
            [
                A.RandomResizedCrop(
                    height=self.hparams.sz,
                    width=self.hparams.sz,
                    scale=(0.7, 1.0),
                ),
                # AdvancedHairAugmentation(),
                A.GridDistortion(),
                A.RandomBrightnessContrast(),
                A.ShiftScaleRotate(),
                A.Flip(p=0.5),
                A.CoarseDropout(
                    max_height=int(self.hparams.sz / 10),
                    max_width=int(self.hparams.sz / 10),
                ),
                # A.HueSaturationValue(),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )
        train_ds = MelanomaDataset(
            df=self.train_df,
            images_path=self.train_images_path,
            augmentations=augmentations,
            train_or_valid=True,
        )
        return DataLoader(
            train_ds,
            # sampler=sampler,
            batch_size=self.hparams.bs,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def forward(self, x):
        x = self.model(x)
        # x = self.head(x).squeeze(1)
        return x

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x).view(-1)
        train_loss = self.loss_function(y_pred=y_pred, y_true=y_true)
        return {
            "loss": train_loss,
            "y_pred": y_pred,
            "y_true": y_true,
        }

    def training_epoch_end(self, outputs: List):
        train_loss = torch.cat(
            [out["loss"].unsqueeze(dim=0) for out in outputs]
        ).mean()
        y_pred = torch.cat([out["y_pred"] for out in outputs], dim=0)
        y_true = torch.cat([out["y_true"] for out in outputs], dim=0)
        train_auc = auroc(y_pred, y_true)

        logs = {"train_loss": train_loss, "train_auc": train_auc}
        return {
            "log": logs,
        }

    def configure_optimizers(self):
        print(f"Using LR: {self.lr}")
        optimizer = Over9000(
            self.parameters(), lr=self.lr, weight_decay=self.hparams.wd,
        )
        cosine_annealing_epochs = int(self.hparams.epochs * 0.60)
        scheduler = DelayedCosineAnnealingLR(
            optimizer,
            delay_epochs=self.hparams.epochs - cosine_annealing_epochs,
            cosine_annealing_epochs=cosine_annealing_epochs,
        )
        return [optimizer], [scheduler]

    def loss_function(self, y_pred, y_true, label_smoothing=0.02):
        y_smo = y_true.float() * (1 - label_smoothing) + 0.5 * label_smoothing

        loss_fn = FocalLoss()
        loss = loss_fn(y_pred, y_smo.type_as(y_pred))
        return loss

    def val_dataloader(self):
        augmentations = Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ]
        )
        valid_ds = MelanomaDataset(
            df=self.valid_df,
            images_path=self.valid_images_path,
            augmentations=augmentations,
            train_or_valid=True,
        )
        return DataLoader(
            valid_ds,
            batch_size=self.hparams.bs,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def test_dataloader(self):
        augmentations = Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ]
        )
        test_ds = MelanomaDataset(
            df=self.test_df,
            images_path=self.test_images_path,
            augmentations=augmentations,  # TODO: add TTA
            train_or_valid=False,
        )
        return DataLoader(
            test_ds,
            batch_size=self.hparams.bs,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x).squeeze().sigmoid()
        loss = self.loss_function(y_pred=y_pred, y_true=y_true)
        return {"val_loss": loss, "y_pred": y_pred, "y_true": y_true}

    def validation_epoch_end(self, outputs: List):
        val_loss = torch.cat(
            [out["val_loss"].unsqueeze(dim=0) for out in outputs]
        ).mean()
        y_pred = torch.cat([out["y_pred"] for out in outputs], dim=0)
        y_true = torch.cat([out["y_true"] for out in outputs], dim=0)
        val_auc = auroc(y_pred, y_true)

        logs = {
            "val_loss": val_loss,
            "val_auc": val_auc,
            "model_name": self.model_name,
            "fold": self.fold,
            "sz": self.sz,
        }
        return {
            "log": logs,
        }

    def test_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x).squeeze().sigmoid()
        return {"y_hat": y_hat}

    def test_epoch_end(self, outputs):
        y_hat = torch.cat([out["y_hat"] for out in outputs])
        self.test_df["preds"] = y_hat.cpu().numpy()
        self.test_df.loc[:, ["image_name", "preds"]].to_csv(
            self.path, index=False
        )
        return {"tta": self.fold}


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


# def split_train_test_using_stratigy_group_k_fold(df):
#     """Split train/valid set."""
#     train_df = df

#     patient_ids = train_df["patient_id"].unique()
#     patient_means = train_df.groupby(["patient_id"])["target"].mean()

#     # split at patient_id level
#     train_idx, val_idx = split_train_test_sets(
#         np.arange(len(patient_ids)),
#         stratify=(patient_means > 0),
#         test_size=0.2,  # validation set size
#     )

#     train_patient_ids = patient_ids[train_idx]
#     train = train_df[train_df.patient_id.isin(train_patient_ids)].reset_index()

#     valid_patient_ids = patient_ids[val_idx]
#     valid = train_df[train_df.patient_id.isin(valid_patient_ids)].reset_index()

#     assert train_df.shape[0] == train.shape[0] + valid.shape[0]

#     return train, valid


if __name__ == "__main__":
    main(create_submission=True)
