import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import pandas as pd
from constants import N_TRAIN


def main():
    train = np.load("./data/x_train_32.npy")
    X_train = train.reshape(N_TRAIN, 32 * 32 * 3)
    y_train = pd.read_csv("./data/train.csv")["target"]

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1, solver="lbfgs", multi_class="multinomial", max_iter=60
        ),
    )
    cv_scores = cross_val_score(
        pipe, X=X_train, y=y_train, cv=10, scoring="roc_auc", n_jobs=-1
    )
    cv_score = np.mean(cv_scores)

    print(np.mean(cv_score))
    with open("./metrics/cv.metric", "w") as f:
        f.write(f"AUC: {cv_score:.4f}")


if __name__ == "__main__":
    main()
