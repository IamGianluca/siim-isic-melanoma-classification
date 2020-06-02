from pathlib import Path

N_TRAIN = 33_126
N_TEST = 10_982

path = Path(".")
data_path = path / "data"
metrics_path = path / "metrics"
models_path = path / "models"
submissions_path = path / "subs"

train_fpath = data_path / "train.csv"
test_fpath = data_path / "test.csv"

train_array_image_fpath = data_path / "x_train_32.npy"
test_array_image_fpath = data_path / "x_test_32.npy"
