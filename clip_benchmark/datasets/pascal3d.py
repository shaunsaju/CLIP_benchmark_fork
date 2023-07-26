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
from typing import Any, Callable, List, Optional, Union, Tuple, Dict
import numpy as np
import random
from imagecorruptions import corrupt
from torchvision.datasets.utils import (
    download_and_extract_archive, verify_str_arg, download_url)


class Pascal3d(VisionDataset):
    """Pascal 3D+ dataset
    """

    def __init__(
            self,
            root: str,
            target_type: Union[List[str], str] = "category",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            split: str = "train",
            corruption: Dict = None

    ) -> None:
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if isinstance(target_type, str):
            target_type = [target_type]
        assert target_type == ["category"]

        if download:
            self.download()
        self.corruption = corruption
        artifact_dir = Path(self.root)
        image_dir = 'PASCAL3D+_release1.1/Images'
        class_folders = [folder for folder in os.listdir(artifact_dir / image_dir) if '_' in folder]
        class_names = list(set([folder.split('_')[0] for folder in class_folders]))
        image_paths = []
        labels = []
        i = 0
        # Iterate through each class folder
        for class_folder in class_folders:
            class_path = os.path.join(artifact_dir, image_dir, class_folder)
            class_index = class_names.index(class_folder.split('_')[0])
            # Get the list of image files
            image_files = os.listdir(class_path)

            # Iterate through each image file
            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)

                # Get the relative path from the current working directory
                relative_path = os.path.relpath(image_path, artifact_dir)

                # Add the image paths and labels
                image_paths.append(relative_path)
                labels.append(class_index)

                i = i + 1

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

        if self.corruption is not None:
            np_image = np.array(img)
            corr = 0
            severity = 1
            if self.corruption['corruption'] == "all":
                corr = random.randint(1, 15)
            if self.corruption['severity'] == "all":
                severity = random.randint(1, 5)
            img = Image.fromarray(corrupt(np_image, severity, corruption_number=corr))

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
        dataset = Pascal3d("/tmp/pascal3d", split=split, download=True)
        print(f"Done loading split {split}: {dataset}")
        print(dataset[0])


if __name__ == "__main__":
    main()
