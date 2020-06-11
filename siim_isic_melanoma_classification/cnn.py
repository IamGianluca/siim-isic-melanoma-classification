import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from pytorch_lightning.core.lightning import LightningModule
from sklearn.metrics import roc_auc_score
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from siim_isic_melanoma_classification.constants import (
    data_path,
    train_img_224_path,
)


class MyModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        # TODO: only train new head
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
                Resize((self.hparams.sz, self.hparams.sz)),
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        train_ds = MelanomaDataset(
            csv_fpath=data_path / "train_patients.csv",
            images_path=train_img_224_path,
            transform=transform,
            train=True,
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
                Resize((self.hparams.sz, self.hparams.sz)),
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        valid_ds = MelanomaDataset(
            csv_fpath=data_path / "valid_patients.csv",
            images_path=train_img_224_path,
            transform=transform,
            train=True,
        )
        return DataLoader(
            valid_ds,
            batch_size=self.hparams.bs,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=False,
        )

    def configure_optimizers(self):
        return SGD(
            self.parameters(), lr=self.hparams.lr, momentum=self.hparams.mom
        )

    def forward(self, x):
        x = self.headless_resnet(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.BCELoss()(y_hat.double(), y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.BCELoss()(y_hat.double().squeeze(), y)
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
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}


class MelanomaDataset(Dataset):
    def __init__(
        self, csv_fpath: Path, images_path: Path, train: bool, transform=None
    ):
        self.melanoma_df = pd.read_csv(csv_fpath, dtype={"target": "float"})
        self.length = self.melanoma_df.shape[0]
        self.images_path = images_path
        self.train = train
        self.image_names = self.melanoma_df.loc[:, "image_name"]
        try:
            self.targets = self.melanoma_df.loc[:, "target"]
        except KeyError:
            self.targets = None
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_fpath = self.images_path / f"{self.image_names[idx]}.jpg"
        image = Image.open(img_fpath)

        if self.transform:
            image = self.transform(image)

        if self.train:
            return image, self.targets[idx]
        else:
            return image
