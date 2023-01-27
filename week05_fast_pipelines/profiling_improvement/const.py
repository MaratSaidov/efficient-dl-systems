import torch


def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class Settings:
    batch_size: int = 64
    epochs: int = 3
    lr: float = 3e-5
    gamma: float = 0.7
    seed: int = 42
    device: str = get_device()


class CatsAndDogs:
    directory = "data"
    train_dir = "data/train"
    test_dir = "data/test"
    regexp = "*.jpg"
