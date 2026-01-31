import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

# ensure `src` is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import data_manager as data


def test_save_and_load_model(tmp_path):
    model = nn.Linear(5, 2)
    save_path = tmp_path / "models" / "tmp_model.pt"

    # save_model should create directories as needed and not raise
    data.save_model(str(save_path), model)
    assert save_path.exists()

    # load_model should return a model with same keys
    loaded = data.load_model(str(save_path), nn.Linear(5, 2))
    assert isinstance(loaded, nn.Module)


def test_mnistdataset_roundtrip(tmp_path):
    # create tiny fake dataset and write it to disk
    images = torch.randint(0, 256, (3, data.TENSOR_SIZE), dtype=torch.uint8)
    labels = torch.tensor([1, 2, 3], dtype=torch.long)

    file_path = tmp_path / "ds.pt"
    torch.save((images, labels), str(file_path))

    ds = data.MNISTDataset(str(file_path))
    assert len(ds) == 3
    x0, y0 = ds[0]
    assert x0.shape[0] == data.TENSOR_SIZE
    assert y0.shape[0] == data.NUM_CLASSES


def test_missing_dataset_raises():
    with pytest.raises(FileNotFoundError):
        data.MNISTDataset("does_not_exist.pt")


def test_save_model_type_error(tmp_path):
    # passing a non-module should raise TypeError
    with pytest.raises(TypeError):
        data.save_model(str(tmp_path / "bad.pt"), object())
