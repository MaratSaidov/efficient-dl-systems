import typing as tp
import glob
import os
import zipfile
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

from const import CatsAndDogs


class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label


def extract_dataset_globs() -> tp.Tuple[tp.List[str], tp.List[str]]:
    """
    TODO: put the description
    """
    os.makedirs(CatsAndDogs.directory, exist_ok=True)

    with zipfile.ZipFile(f"{CatsAndDogs.train_dir}.zip") as train_zip:
        train_zip.extractall(CatsAndDogs.directory)
    
    with zipfile.ZipFile(f"{CatsAndDogs.test_dir}.zip") as test_zip:
        test_zip.extractall(CatsAndDogs.directory)

    train_list = glob.glob(os.path.join(CatsAndDogs.train_dir, CatsAndDogs.regexp))
    test_list = glob.glob(os.path.join(CatsAndDogs.test_dir, CatsAndDogs.regexp))
    
    return train_list, test_list


def get_labels(globs_list: tp.List[str]) -> tp.List[str]:
    return [path.split('/')[-1].split('.')[0] for path in globs_list]


def get_train_transforms() -> tp.Any:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


def get_test_transforms() -> tp.Any:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
