import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from albumentations.augmentations.transforms import Flip, Normalize, Resize
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensor
from PIL import Image
from pytorch_lightning.core.lightning import LightningModule
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from siim_isic_melanoma_classification.constants import train_img_224_path


class MyModel(LightningModule):
    def __init__(
        self, hparams, train_df: pd.DataFrame, valid_df: pd.DataFrame
    ):
        super().__init__()
        self.hparams = hparams
        self.train_df = train_df
        self.valid_df = valid_df

        if "resnet" not in self.hparams.arch:
            raise ValueError(
                "Not tested for architectures different from ResNet"
            )
        original_model = models.__dict__[self.hparams.arch](pretrained=True)
        image_modules = list(original_model.children())[
            :-1
        ]  # keep all layers except last one
        self.headless_resnet = nn.Sequential(*image_modules)
        self.fc = nn.Linear(
            in_features=original_model.fc.in_features,
            out_features=1,
            bias=True,
        )

    def train_dataloader(self):
        transform = Compose(
            [
                Resize(height=self.hparams.sz, width=self.hparams.sz),
                Flip(),
                Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ToTensor(),
            ]
        )
        train_ds = MelanomaDataset(
            df=self.train_df,
            images_path=train_img_224_path,
            transform=transform,
            train_or_valid=True,
        )
        return DataLoader(
            train_ds,
            batch_size=self.hparams.bs,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=False,
        )

    def val_dataloader(self):
        transform = Compose(
            [
                Resize(height=self.hparams.sz, width=self.hparams.sz),
                Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ToTensor(),
            ]
        )
        valid_ds = MelanomaDataset(
            df=self.valid_df,
            images_path=train_img_224_path,
            transform=transform,
            train_or_valid=True,
        )
        return DataLoader(
            valid_ds,
            batch_size=self.hparams.bs,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=False,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=3e-6
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=10 * self.hparams.lr,
            epochs=self.hparams.epochs,
            steps_per_epoch=len(self.train_dataloader()),
        )
        return [optimizer], [scheduler]

    def forward(self, x):
        x = self.headless_resnet(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.BCELoss()(y_hat.double(), y.double())
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # something is wrong here
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.BCELoss()(y_hat.double().squeeze(), y.double())
        return {"val_loss": loss, "probs": y_hat, "gt": y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.cat(
            [out["val_loss"].unsqueeze(dim=0) for out in outputs]
        ).mean()
        probs = torch.cat([out["probs"] for out in outputs], dim=0)
        gt = torch.cat([out["gt"] for out in outputs], dim=0)

        # move to CPU since we are using sklearn function to compute AUC:w
        probs = probs.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()

        auc_roc = torch.tensor(roc_auc_score(gt, probs))
        tensorboard_logs = {"val_loss": avg_loss, "auc": auc_roc}
        print(
            f"Epoch {self.current_epoch}: {avg_loss:.2f}, auc: {auc_roc:.4f}"
        )
        return {
            "avg_val_loss": avg_loss,
            "log": tensorboard_logs,
        }


class MelanomaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_path: Path,
        train_or_valid: bool,
        transform=None,
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
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_fpath = self.images_path / f"{self.image_names[idx]}.jpg"
        image = np.array(Image.open(img_fpath))

        if self.transform:
            image = self.transform(image=image)["image"]

        if self.train_or_valid:
            return image, self.targets[idx]
        else:
            return image
