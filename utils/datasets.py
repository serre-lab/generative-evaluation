import os
import scipy.io
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from typing import Optional, Callable


class ArcaroDataset(Dataset):
    def __init__(
        self, root: str, monkey: str = "Red", transform: Optional[Callable] = None
    ):
        self.root = root
        self.transform = transform
        self.monkey_root = os.path.join(root, monkey)
        if monkey == "Red":
            self.data_path = os.path.join(self.monkey_root, "red_062419_data.mat")
            self.im_dir = os.path.join(self.monkey_root, "red_naturalface")
        elif monkey == "George":
            self.data_path = os.path.join(self.monkey_root, "george_060118_data.mat")
            self.im_dir = os.path.join(self.monkey_root, "natural_test")
        else:
            raise ValueError(f"Unsupported monkey: {monkey}")

        mat = scipy.io.loadmat(self.data_path)
        self.paths = [str(name[0]) for name in mat["UniqueNames"][0]]
        self.responses = mat["imageRF"]
        assert self.responses.shape[-1] == len(
            self.paths
        ), "Mismatch between responses and paths length"

    def __len__(self):
        # Return the size of the dataset
        return len(self.paths)

    def __getitem__(self, idx):
        # Get an item from the dataset
        path = os.path.join(self.im_dir, self.paths[idx])
        im = Image.open(path).convert("RGB")
        response = self.responses[:, :, :, :, idx]

        if self.transform:
            im = self.transform(im)

        return im, response
