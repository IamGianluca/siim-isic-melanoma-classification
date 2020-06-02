import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from constants import (
    N_TRAIN,
    metrics_path,
    models_path,
    train_array_image_fpath,
    train_fpath,
)


def main():
    train = np.load(train_array_image_fpath)
    X_train = train.reshape(N_TRAIN, 32 * 32 * 3)
    y_train = pd.read_csv(train_fpath)["target"]

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1, solver="lbfgs", multi_class="multinomial", max_iter=60
        ),
    )

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
    joblib.dump(pipe, models_path / "stage1.joblib")


if __name__ == "__main__":
    main()
