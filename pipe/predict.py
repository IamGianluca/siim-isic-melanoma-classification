import joblib
import numpy as np
import pandas as pd

from constants import (
    N_TEST,
    test_array_image_fpath,
    submissions_path,
    test_fpath,
    models_path,
)


def main():
    ar = np.load(test_array_image_fpath)
    X_test = ar.reshape(N_TEST, 32 * 32 * 3)

    pipe = joblib.load(models_path / "stage1.joblib")
    y_preds = pipe.predict_proba(X_test)

    image_names = pd.read_csv(test_fpath)
    preds_df = pd.DataFrame(
        {"image_name": image_names["image_name"], "target": y_preds[:, 1]}
    )
    preds_df.to_csv(submissions_path / "submission.csv", index=False)


if __name__ == "__main__":
    main()
