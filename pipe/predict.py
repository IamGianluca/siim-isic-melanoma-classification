import joblib

from siim_isic_melanoma_classification.constants import models_path
from siim_isic_melanoma_classification.prepare import prepare_dataset
from siim_isic_melanoma_classification.submit import prepare_submission


def main():
    X_test, _ = prepare_dataset(name="test")

    sclf = joblib.load(models_path / "stage1_lr.joblib")
    y_preds = sclf.predict_proba(X_test)

    # prepare submission
    prepare_submission(y_preds=y_preds[:, 1], fname="lr_submission.csv")


if __name__ == "__main__":
    main()
