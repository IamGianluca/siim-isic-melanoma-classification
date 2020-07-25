import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision.utils as utils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.saving import load_hparams_from_yaml
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from siim_isic_melanoma_classification.cnn import MyModel
from siim_isic_melanoma_classification.constants import (
    data_path,
    folds_fpath,
    metrics_path,
    models_path,
    params_fpath,
    submissions_path,
    test_fpath,
    test_img_128_path,
    train_img_128_path,
)
from siim_isic_melanoma_classification.utils import dict_to_args

params = load_hparams_from_yaml(params_fpath)
hparams = dict_to_args(params["train_resnet_128"])
logger = MLFlowLogger("logs/")

name = " ".join(re.findall("[a-zA-Z]+", hparams.arch))

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
    predictions.to_csv(submission_fpath, index=False)


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


def split_train_test_sets(folds, fold_number):
    train_df = folds[folds.fold != fold_number].reset_index(drop=True)
    test_df = folds[folds.fold == fold_number].reset_index(drop=True)
    return train_df, test_df


def train(folds: pd.DataFrame, fold_number: int, path):
    """Create OOF predictions for fold_number."""
    train_df, test_df = split_train_test_sets(folds, fold_number)

    # train model and report valid scores progress during training
    model = MyModel(
        hparams=hparams,
        fold=fold_number,
        train_df=train_df,
        valid_df=test_df,
        test_df=test_df,
        train_images_path=train_img_128_path,
        valid_images_path=train_img_128_path,
        test_images_path=train_img_128_path,  # NOTE: OOF predictions
        path=path,
    )
    callback = ModelCheckpoint(
        filepath=models_path,
        monitor="val_auc",
        mode="max",
        save_weights_only=True,
    )

    print("O2" if hparams.precision == 32 else "O1")
    trainer = Trainer(
        gpus=1,
        max_epochs=hparams.epochs,
        progress_bar_refresh_rate=0,
        # auto_lr_find=True,
        # overfit_batches=5,
        amp_level="O2" if hparams.precision == 32 else "O1",
        precision=hparams.precision,
        logger=logger,
        checkpoint_callback=callback,
    )
    # # Run learning rate finder
    # lr_finder = trainer.lr_find(model)
    # new_lr = lr_finder.suggestion()
    # print(new_lr)
    # model.hparams.lr = new_lr

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
    model.test_images_path = test_img_128_path

    results = list()
    for batch in model.test_dataloader():
        batch = batch.to("cuda")
        results.append(model(batch).cpu().numpy().reshape(-1))
    preds = np.concatenate(results, axis=0)
    preds_df = pd.DataFrame({"image_name": test_df.image_name, "preds": preds})
    return preds_df


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


if __name__ == "__main__":
    main(create_submission=True)
