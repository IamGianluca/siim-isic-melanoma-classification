from pathlib import Path

N_TRAIN = 33_126
N_TEST = 10_982

path = Path("/home/gianluca/git/kaggle/siim-isic-melanoma-classification")

params_fpath = path / "params.yaml"

data_path = path / "data"
train_img_path = data_path / "train"
test_img_path = data_path / "test"

train_img_224_path = data_path / "train_224"
test_img_224_path = data_path / "test_224"

metrics_path = path / "metrics"
models_path = path / "models"
submissions_path = path / "subs"

train_fpath = data_path / "train.csv"
test_fpath = data_path / "test.csv"
dupes_fpath = data_path / "dupes.csv"
folds_fpath = data_path / "folds.csv"

train_array_image_fpath = data_path / "x_train_32.npy"
test_array_image_fpath = data_path / "x_test_32.npy"
