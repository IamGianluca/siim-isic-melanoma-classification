import joblib
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from siim_isic_melanoma_classification.constants import (
    metrics_path,
    models_path,
)
from siim_isic_melanoma_classification.prepare import prepare_dataset
from siim_isic_melanoma_classification.submit import prepare_submission


def main():
    X_train, y_train = prepare_dataset(name="train")

    pixel_cols = list(range(32 * 32 * 3))

    tfms = make_column_transformer(
        # flattened image data
        (StandardScaler(), pixel_cols),
        # other contextual features
        (
            SimpleImputer(
                missing_values=np.NaN,
                strategy="constant",
                fill_value=0,
                add_indicator=True,
            ),
            ["age_approx"],
        ),
        (
            make_pipeline(
                SimpleImputer(
                    missing_values=np.NaN,
                    strategy="most_frequent",
                    add_indicator=True,  # does this column get feed into OHE?
                ),
                OneHotEncoder(handle_unknown="ignore"),
            ),
            ["sex", "anatom_site_general_challenge"],
        ),
        remainder="drop",
    )
    clf = LogisticRegression(
        C=1, solver="lbfgs", multi_class="multinomial", max_iter=60
    )
    pipe = make_pipeline(tfms, clf)

    # cross validation
    cv_results = cross_validate(
        pipe, X=X_train, y=y_train, cv=10, scoring="roc_auc", n_jobs=-1,
    )
    cv_score = np.mean(cv_results["test_score"])

    print(f"CV AUC: {cv_score:.4f}")
    with open(metrics_path / "cv.metric", "w") as f:
        f.write(f"AUC: {cv_score:.4f}")

    # train model on entire training dataset
    pipe.fit(X=X_train, y=y_train)
    joblib.dump(pipe, models_path / "stage1_lr.joblib")

    # predict on test data
    X_test, _ = prepare_dataset(name="test")

    sclf = joblib.load(models_path / "stage1_lr.joblib")
    y_preds = sclf.predict_proba(X_test)

    # prepare submission
    prepare_submission(y_preds=y_preds[:, 1], fname="lr_submission.csv")


if __name__ == "__main__":
    main()
