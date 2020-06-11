import argparse
from typing import List

import matplotlib.pyplot as plt
import torch
import torchvision.utils as utils
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from siim_isic_melanoma_classification.cnn import MelanomaDataset, MyModel
from siim_isic_melanoma_classification.constants import (
    data_path,
    models_path,
    test_img_224_path,
)
from siim_isic_melanoma_classification.submit import prepare_submission

model_fpath = models_path / "stage1_resnet18.pth"


arch = "resnet18"
sz = 128
bs = 600
lr = 1e-4
mom = 0.9
epochs = 80


def main(train: bool = True):
    hparams = dict_to_args(
        {"arch": arch, "sz": sz, "bs": bs, "lr": lr, "mom": mom}
    )
    model = MyModel(hparams=hparams)
    # from pytorch_lightning.profiler import AdvancedProfiler

    # profiler = AdvancedProfiler()
    trainer = Trainer(
        gpus=1,
        max_epochs=epochs,  # train_percent_check=0.1, profiler=profiler
    )

    if train:
        print("Train classifier...")
        trainer.fit(model)

        print("Persisting model to disk...")
        trainer.save_checkpoint(model_fpath)

    print("Making predictions on test data...")
    model.load_from_checkpoint(model_fpath)
    model.freeze()

    # TODO: move this inside Model and maybe find a way to keep it in sync with
    # the transformations we make on the validation set
    transform = Compose(
        [
            Resize((128, 128)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_ds = MelanomaDataset(
        csv_fpath=data_path / "test.csv",
        images_path=test_img_224_path,
        transform=transform,
        train=False,
    )
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=8,)
    results: List[torch.Tensor] = list()
    for batch_num, data in enumerate(test_dl):
        data = data.to("cuda")
        results.append(model(data))
    y_preds = torch.cat(results, axis=0)

    prepare_submission(
        y_preds=y_preds.cpu().numpy().squeeze(),
        fname=f"{arch}_submission.csv",
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


if __name__ == "__main__":
    main(train=True)
