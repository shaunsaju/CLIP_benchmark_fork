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


class Caltech101Vic(VisionDataset):
    """Modified Caltech101
    annotations from VIC
        https://github.com/altndrr/vic
    data from torchvision dataset
        https://github.com/pytorch/vision/blob/main/torchvision/datasets/caltech.py

    Statistics:
        - Around 9,000 images.
        - 100 classes.
        - URL: https://data.caltech.edu/records/mzrjq-6wc02.

    Reference:
        - Li et al. One-shot learning of object categories. TPAMI 2006.

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
        super().__init__(os.path.join(root, "caltech101"), transform=transform,
                         target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if isinstance(target_type, str):
            target_type = [target_type]
        assert target_type == ["category"]

        if download:
            self.download()

        artifact_dir = Path(self.root)
        metadata_fp = artifact_dir / "metadata.csv"
        split_fp = artifact_dir / "split_coop.csv"

        metadata_df = pd.read_csv(metadata_fp)
        class_names = metadata_df["class_name"].tolist()
        classes_to_idx = {str(c): i for i, c in enumerate(metadata_df["folder_name"].tolist())}

        split_df = pd.read_csv(split_fp)
        dataset_split = split
        assert dataset_split in ["train", "val", "test"]
        image_paths = split_df[split_df["split"] == dataset_split]["filename"]
        image_paths = image_paths.apply(lambda x: str(Path("101_ObjectCategories") / x)).tolist()
        folder_names = [Path(f).parent.name for f in image_paths]
        labels = [classes_to_idx[c] for c in folder_names]

        self.classes = class_names
        self.targets = labels
        self.imgs = image_paths

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
        # download data using torchvision
        if not os.path.exists(os.path.join(self.root, "101_ObjectCategories")):
            download_and_extract_archive(
                "https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp",
                self.root,
                filename="101_ObjectCategories.tar.gz",
                md5="b224c7392d521a49829488ab0f1120d9",
            )
        if not os.path.exists(os.path.join(self.root, "Annotations")):
            download_and_extract_archive(
                "https://drive.google.com/file/d/175kQy3UsZ0wUEHZjqkUDdNVssr7bgh_m",
                self.root,
                filename="Annotations.tar",
                md5="6f83eeb1f24d99cab4eb377263132c91",
            )

        if not os.path.exists(os.path.join(self.root, "metadata.csv")):
            download_url(
                "https://raw.githubusercontent.com/altndrr/vic/main/artifacts/data/caltech101/metadata.csv",
                self.root)
        if not os.path.exists(os.path.join(self.root, "split_coop.csv")):
            download_url(
                "https://raw.githubusercontent.com/altndrr/vic/main/artifacts/data/caltech101/split_coop.csv",
                self.root)


def main():
    for split in ["train", "val", "test"]:
        dataset = Caltech101Vic("/tmp/caltech", split=split, download=True)
        print(f"Done loading split {split}: {dataset}")
        print(dataset[0])


if __name__ == "__main__":
    main()
