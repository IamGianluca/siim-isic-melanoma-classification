import joblib
import pandas as pd

from siim_isic_melanoma_classification.constants import (
    models_path,
    submissions_path,
    test_fpath,
)
from siim_isic_melanoma_classification.prepare import prepare_dataset


def main():
    X_test, _ = prepare_dataset(name="test")

    sclf = joblib.load(models_path / "stage1_lr.joblib")
    y_preds = sclf.predict_proba(X_test)

    # prepare submission
    prepare_submission(y_preds=y_preds)


def prepare_submission(y_preds):
    image_names = pd.read_csv(test_fpath)
    preds_df = pd.DataFrame(
        {"image_name": image_names["image_name"], "target": y_preds[:, 1]}
    )
    preds_df.to_csv(submissions_path / "submission.csv", index=False)


if __name__ == "__main__":
    main()
