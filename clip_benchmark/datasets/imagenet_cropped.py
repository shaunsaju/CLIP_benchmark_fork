import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import check_integrity, verify_str_arg

META_FILE = "meta.bin"


class ImageNetCropped(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root: str, subfolder: str, split: str = "train", **kwargs: Any) -> None:
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))
        self.subfolder = subfolder

        self.parse_archives()
        wnid_to_classes = load_meta_file(self.root)[0]

        super().__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}
        print(f"Will load images from {self.split_folder}")

    def parse_archives(self) -> None:
        assert (Path(self.root) / META_FILE).is_file(), f"Path {self.root} missing"
        assert os.path.isdir(self.split_folder), f"Path {self.split_folder} missing"

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, f"{self.subfolder}/{self.split}")

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


def load_meta_file(root: str, file: Optional[str] = None) -> Tuple[Dict[str, str], List[str]]:
    if file is None:
        file = META_FILE
    file = os.path.join(root, file)

    if check_integrity(file):
        return torch.load(file)
    else:
        msg = (
            "The meta file {} is not present in the root directory or is corrupted. "
            "This file is automatically created by the ImageNet dataset."
        )
        raise RuntimeError(msg.format(file, root))
