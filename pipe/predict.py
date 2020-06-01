import joblib
import pandas as pd
from constants import N_TEST
import numpy as np


def main():
    ar = np.load("./data/x_test_32.npy")
    X_test = ar.reshape(N_TEST, 32 * 32 * 3)

    pipe = joblib.load("./models/stage1.joblib")
    y_preds = pipe.predict_proba(X_test)

    image_names = pd.read_csv("./data/test.csv")
    preds_df = pd.DataFrame(
        {"image_name": image_names["image_name"], "target": y_preds[:, 1]}
    )
    preds_df.to_csv("./subs/submission.csv", index=False)


if __name__ == "__main__":
    main()
