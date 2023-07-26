import os
import tarfile
import zipfile
from math import ceil
from pathlib import Path
from shutil import rmtree

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, List, Optional, Union, Tuple

from torchvision.datasets.utils import (
    download_and_extract_archive, verify_str_arg, download_url)


class OODCV(VisionDataset):
    """OODCV dataset
    """

    def __init__(
            self,
            root: str,
            target_type: Union[List[str], str] = "category",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            split: str = "train",
    ) -> None:
        super().__init__(os.path.join(root, "pascal3d"), transform=transform,
                         target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if isinstance(target_type, str):
            target_type = [target_type]
        assert target_type == ["category"]

        if download:
            self.download()

        artifact_dir = Path(self.root)
        image_dir = 'train/Images'
        labels_csv_path = "train/labels.csv"
        labels_df = pd.read_csv(artifact_dir/ labels_csv_path)
        class_names = list(set(list(labels_df['labels'])))

        def mod_class_idx(row):
            row['labels'] = class_names.index(row['labels'])
            folder = row['imgs'].split('_')[1]
            row['imgs'] = os.path.join(artifact_dir, image_dir, folder, row['imgs'])
            return row

        labels_df = labels_df.apply(mod_class_idx, axis=1)

        self.classes = class_names
        self.targets = list(labels_df['labels'])
        self.imgs = list(labels_df['imgs'])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        img = Image.open(
            os.path.join(self.root, self.imgs[index]))

        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.imgs)

    def download(self) -> None:
        NotImplementedError()


def main():
    for split in ["train", "val", "test"]:
        dataset = OODCV("/tmp/oodcv", split=split, download=True)
        print(f"Done loading split {split}: {dataset}")
        print(dataset[0])


if __name__ == "__main__":
    main()
