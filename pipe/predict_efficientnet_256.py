import numpy as np
import pandas as pd
import albumentations.augmentations.transforms as A
from pytorch_lightning.core.saving import load_hparams_from_yaml
from albumentations.core.composition import Compose
from albumentations.pytorch.transforms import ToTensorV2

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

from pipe.train_efficientnet_256 import (
    MelanomaDataset,
    MyModel,
    sort_oof_predictions,
    split_train_test_sets,
)
from siim_isic_melanoma_classification.utils import dict_to_args
from torch.utils.data.dataloader import DataLoader
import os
from sklearn.metrics import roc_auc_score

params = load_hparams_from_yaml(params_fpath)
hparams = dict_to_args(params["train_efficientnet_256"])

name = "efficientnet"
metric_fpath = metrics_path / "l1_efficientnet_tta_256_cv.metric"


def main(tta_iters=50):
    """Validate with OOF predictions the effectiveness of using different
    levels of TTA.
    """
    folds = pd.read_csv(folds_fpath)
    n_folds = folds.fold.nunique()

    final = list()
    for fold_number in range(n_folds):
        train_df, valid_df = split_train_test_sets(
            folds, fold_number, use_extra_data=True
        )

        # load pretrained model
        ckpt_fpath = (
            models_path
            / f"model_name=efficientnet_sz=256_fold={fold_number}.ckpt"
        )
        model = MyModel.load_from_checkpoint(checkpoint_path=str(ckpt_fpath))
        model.freeze()
        model.to("cuda")

        # setup ds and dl
        valid_ds = MelanomaDataset(
            df=valid_df,
            images_path=train_img_256_path,  # we are doing OOF CV
            augmentations=get_tta_transforms(),
            train_or_valid=True,
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=hparams.bs * 3,
            shuffle=False,  # important for the order of the OOF preds
            num_workers=os.cpu_count() * 2,
            pin_memory=True,
        )

        # setup TTA
        model.test_df = valid_df
        # model.test_images_path = train_img_256_path
        model.test_dataloader = lambda: valid_dl

        # save OOF predictions
        tta_fold_preds = list()
        for _ in range(tta_iters):
            results = list()
            for batch in model.test_dataloader():
                img, _ = batch
                img = img.to("cuda")
                results.append(model(img).cpu().numpy().reshape(-1))
            oof_preds = np.concatenate(results, axis=0)

            tta_fold_preds.append(
                pd.DataFrame(
                    {"image_name": valid_df.image_name, "preds": oof_preds}
                )
            )

        tta_preds = pd.concat([df.preds for df in tta_fold_preds], axis=1)
        tta_preds.columns = [f"preds_ttae{n}" for n in range(tta_iters)]
        tta_preds["image_name"] = valid_df.image_name

        final.append(tta_preds)

    final_concat = pd.concat(final, axis=0)
    oof_preds = final_concat.set_index("image_name").mean(axis=1)
    oof_preds = oof_preds.reset_index()
    oof_preds.columns = ["image_name", "preds"]

    # save oof preds for stacking
    y_true = folds.loc[:, ["image_name", "target"]]
    sorted_oof_preds = pd.merge(
        y_true,
        oof_preds,
        how="inner",
        left_on="image_name",
        right_on="image_name",
    )

    # oof cv score
    cv_score = roc_auc_score(
        sorted_oof_preds["target"], sorted_oof_preds["preds"]
    )
    print(f"OOF CV AUC:{cv_score:.4f}")
    with open(metric_fpath, "w") as f:
        f.write(f"OOF CV AUC: {cv_score:.4f}")
    oof_preds.to_csv(
        data_path / "oof_preds_efficientnet_tta_256.csv", index=False
    )

    ###############################################
    # PREDICT ON TEST DATA
    ###############################################
    final = list()
    for fold_number in range(n_folds):
        # load pretrained model
        ckpt_fpath = (
            models_path
            / f"model_name=efficientnet_sz=256_fold={fold_number}.ckpt"
        )
        model = MyModel.load_from_checkpoint(checkpoint_path=str(ckpt_fpath))
        model.freeze()
        model.to("cuda")

        # setup ds and dl
        test_df = pd.read_csv(test_fpath)
        test_ds = MelanomaDataset(
            df=test_df,
            images_path=test_img_256_path,  # we are doing OOF CV
            augmentations=get_tta_transforms(),
            train_or_valid=False,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=hparams.bs * 3,
            shuffle=False,  # important for the order of the OOF preds
            num_workers=os.cpu_count() * 2,
            pin_memory=True,
        )

        # setup TTA
        model.test_df = test_df
        # model.test_images_path = test_img_256_path
        model.test_dataloader = lambda: test_dl

        # save OOF predictions
        tta_fold_preds = list()
        for _ in range(tta_iters):
            results = list()
            for batch in model.test_dataloader():
                img = batch
                img = img.to("cuda")
                results.append(model(img).cpu().numpy().reshape(-1))
            oof_preds = np.concatenate(results, axis=0)

            tta_fold_preds.append(
                pd.DataFrame(
                    {"image_name": test_df.image_name, "preds": oof_preds}
                )
            )

        tta_preds = pd.concat([df.preds for df in tta_fold_preds], axis=1)
        tta_preds.columns = [f"preds_tta{n}" for n in range(tta_iters)]
        tta_preds["image_name"] = test_df.image_name

        final.append(tta_preds)
    final_concat = pd.concat(
        [f.set_index("image_name") for f in final], axis=1
    )
    preds = final_concat.mean(axis=1)
    preds = preds.reset_index()
    preds.columns = ["image_name", "target"]
    preds.to_csv(
        submissions_path / "l1_efficientnet_tta_256_submission.csv",
        index=False,
    )


def get_tta_transforms():
    return Compose(
        [
            A.RandomResizedCrop(
                height=hparams.sz, width=hparams.sz, scale=(0.7, 1.0),
            ),
            # AdvancedHairAugmentation(),
            A.GridDistortion(),
            A.RandomBrightnessContrast(),
            A.ShiftScaleRotate(),
            A.Flip(p=0.5),
            A.CoarseDropout(
                max_height=int(hparams.sz / 10),
                max_width=int(hparams.sz / 10),
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


if __name__ == "__main__":
    main()
