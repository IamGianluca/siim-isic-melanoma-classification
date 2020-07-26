from pathlib import Path

import numpy as np
import pandas as pd

from siim_isic_melanoma_classification.constants import test_fpath


def prepare_submission(y_pred: np.ndarray, fpath: Path):
    image_names = pd.read_csv(test_fpath)
    preds_df = pd.DataFrame(
        {"image_name": image_names["image_name"], "target": y_pred}
    )
    preds_df.to_csv(fpath, index=False)
