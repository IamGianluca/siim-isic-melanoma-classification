import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from albumentations.augmentations.transforms import (
    CoarseDropout,
    Flip,
    GridDistortion,
    Normalize,
    RandomBrightnessContrast,
    RandomResizedCrop,
    ShiftScaleRotate,
)
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from PIL import Image
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.functional import auroc
from torch.utils.data import DataLoader, Dataset

from siim_isic_melanoma_classification.lr_scheduler import (
    DelayedCosineAnnealingLR,
)
from siim_isic_melanoma_classification.over9000 import Over9000


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        avg_x = self.avg(x)
        max_x = self.max(x)
        return torch.cat([avg_x, max_x], dim=1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class MyModel(LightningModule):
    def __init__(
        self,
        hparams,
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
        self.hparams = hparams
        self.fold = fold

        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

        self.train_images_path = train_images_path
        self.valid_images_path = valid_images_path
        self.test_images_path = test_images_path

        if "resnet" in self.hparams.arch:
            self.model = models.__dict__[self.hparams.arch](pretrained=True)
            c_feature = self.model.fc.in_features
            remove_range = 2  # TODO: try only removing last layer
        if "resnext" in self.hparams.arch:
            self.model = torch.hub.load(
                "facebookresearch/semi-supervised-ImageNet1K-models",
                self.hparams.arch,
            )
            c_feature = self.model.fc.in_features
            remove_range = 2  # TODO: ditto
        if "efficient" in self.hparams.arch:
            self.model = EfficientNet.from_pretrained(
                self.hparams.arch, advprop=True
            )
        c_out = 1
        # self.model = nn.Sequential(
        #     *list(self.model.children())[:-remove_range]
        # )
        self.head = nn.Sequential(
            nn.ReLU(), nn.Dropout(), nn.Linear(1000, c_out)
        )

    def train_dataloader(self):
        augmentations = Compose(
            [
                RandomResizedCrop(
                    height=self.hparams.sz,
                    width=self.hparams.sz,
                    scale=(0.7, 1.0),
                ),
                #         ToGray(),
                GridDistortion(),
                RandomBrightnessContrast(),
                ShiftScaleRotate(),
                Flip(p=0.5),
                CoarseDropout(
                    max_height=int(self.hparams.sz / 10),
                    max_width=int(self.hparams.sz / 10),
                ),
                Normalize(
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
            pin_memory=False,
        )

    def forward(self, x):
        x = self.model(x)
        x = self.head(x).squeeze(1)
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
        optimizer = Over9000(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
        )
        cosine_annealing_epochs = int(self.hparams.epochs * 0.20)
        scheduler = DelayedCosineAnnealingLR(
            optimizer,
            delay_epochs=self.hparams.epochs - cosine_annealing_epochs,
            cosine_annealing_epochs=cosine_annealing_epochs,
        )
        return [optimizer], [scheduler]

    def loss_function(self, y_pred, y_true):
        loss_fn = FocalLoss()
        y_true = y_true.float()
        loss = loss_fn(y_pred, y_true)
        return loss

    def val_dataloader(self):
        augmentations = Compose(
            [
                Normalize(
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
            pin_memory=False,
        )

    def test_dataloader(self):
        augmentations = Compose(
            [
                Normalize(
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
            pin_memory=False,
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

        logs = {"val_loss": val_loss, "val_auc": val_auc}
        print(
            f"Epoch {self.current_epoch} // val loss: {val_loss:.4f}, val auc: {val_auc:.4f}, pos: {y_true.sum()}, neg: {len(y_true) - y_true.sum()}"
        )
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class MelanomaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_path: Path,
        train_or_valid: bool,
        augmentations=None,
    ):
        self.df = df
        self.length = self.df.shape[0]
        self.images_path = images_path
        self.train_or_valid = train_or_valid
        self.image_names = self.df.loc[:, "image_name"]
        try:
            self.targets = self.df.loc[:, "target"]
        except KeyError:
            self.targets = None
        self.augmentations = augmentations

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        img_fpath = self.images_path / f"{self.image_names[item]}.jpg"
        image = np.array(Image.open(img_fpath))

        if self.augmentations:
            image = self.augmentations(image=image)["image"]

        if self.train_or_valid:
            return image, self.targets[item]
        else:
            return image
