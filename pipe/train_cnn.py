from typing import List
from pytorch_lightning import Trainer
import torchvision.utils as utils

import matplotlib.pyplot as plt
import torch
from siim_isic_melanoma_classification.submit import prepare_submission
from siim_isic_melanoma_classification.cnn import MyModel
from siim_isic_melanoma_classification.constants import models_path
from siim_isic_melanoma_classification.cnn import MelanomaDataset
from torch.utils.data import DataLoader
from siim_isic_melanoma_classification.constants import (
    data_path,
    test_img_224_path,
)
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import argparse

model_fpath = models_path / "stage1_resnet18.pth"


def main(train: bool = True, epochs: int = 5, bs: int = 200, sz: int = 128):
    transform = Compose(
        [Resize((128, 128)), ToTensor(), Normalize(mean=0, std=1)]
    )
    print("Creating Dataset")
    test_ds = MelanomaDataset(
        csv_fpath=data_path / "test.csv",
        images_path=test_img_224_path,
        transform=transform,
        train=False,
    )
    print("Creating DataLoaders...")
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=8,)
    # plot_a_batch(train_dl)

    print("Setting up model...")
    hparams = dict_to_args({"sz": 128, "bs": 400, "lr": 1e-4, "mom": 0.9})
    model = MyModel(hparams=hparams)
    # from pytorch_lightning.profiler import AdvancedProfiler

    # profiler = AdvancedProfiler()
    trainer = Trainer(
        gpus=1, max_epochs=80,  # train_percent_check=0.1, profiler=profiler
    )

    if train:
        print("Train classifier...")
        trainer.fit(model)

        print("Persisting model to disk...")
        trainer.save_checkpoint(model_fpath)

    print("Making predictions on test data...")
    model.load_from_checkpoint(model_fpath)
    model.freeze()

    results: List[torch.Tensor] = list()
    for batch_num, data in enumerate(test_dl):
        data = data.to("cuda")
        results.append(model(data))
    y_preds = torch.cat(results, axis=0)
    prepare_submission(
        y_preds=y_preds.cpu().numpy().squeeze(),
        fname="resnet18_submission.csv",
    )


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


def plot_a_batch(train_dl):
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


if __name__ == "__main__":
    main(train=True)
