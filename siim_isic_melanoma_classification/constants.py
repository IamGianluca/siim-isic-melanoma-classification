from pathlib import Path

N_TRAIN = 33_126
N_TEST = 10_982

path = Path("/home/gianluca/git/kaggle/siim-isic-melanoma-classification")

params_fpath = path / "params.yaml"

data_path = path / "data"
train_img_path = data_path / "train"
test_img_path = data_path / "test"

train_img_128_path = data_path / "train_128"
train_img_128_extra_path = data_path / "train_with_extra_128"
test_img_128_path = data_path / "test_128"
train_img_192_path = data_path / "train_192"
train_img_192_extra_path = data_path / "train_with_extra_192"
test_img_192_path = data_path / "test_192"
train_img_256_path = data_path / "train_256"
train_img_256_extra_path = data_path / "train_with_extra_256"
test_img_256_path = data_path / "test_256"
train_img_384_path = data_path / "train_384"
train_img_384_extra_path = data_path / "train_with_extra_384"
test_img_384_path = data_path / "test_384"
train_img_512_path = data_path / "train_512"
train_img_512_extra_path = data_path / "train_with_extra_512"
test_img_512_path = data_path / "test_512"
train_img_768_path = data_path / "train_768"
test_img_768_path = data_path / "test_768"
train_img_1024_path = data_path / "train_1024"
test_img_1024_path = data_path / "test_1024"

metrics_path = path / "metrics"
models_path = path / "models"
submissions_path = path / "subs"

train_fpath = data_path / "train.csv"
test_fpath = data_path / "test.csv"
dupes_fpath = data_path / "dupes.csv"
folds_fpath = data_path / "folds.csv"
folds_with_extra_fpath = data_path / "folds_with_extra.csv"

train_array_image_fpath = data_path / "x_train_32.npy"
test_array_image_fpath = data_path / "x_test_32.npy"
