import numpy as np
import pandas as pd

from siim_isic_melanoma_classification.constants import (
    submissions_path,
    test_fpath,
)


def prepare_submission(y_preds: np.ndarray, fname: str):
    image_names = pd.read_csv(test_fpath)
    preds_df = pd.DataFrame(
        {"image_name": image_names["image_name"], "target": y_preds}
    )
    preds_df.to_csv(submissions_path / fname, index=False)
