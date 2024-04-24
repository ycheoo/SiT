import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torchvision.datasets.folder import DatasetFolder


def random_crop_resize(sample, crop_minlen, input_size):
    # support for dual channels
    if sample.ndim == 1 or sample.shape[0] == 1:
        sample = np.vstack([sample, sample])

    # if sample is too short (less than crop_minlen)
    if sample.shape[1] < crop_minlen:
        sample_padded = np.zeros((2, crop_minlen), dtype=np.float32)
        sample_padded[:, sample.shape[1]] = sample
        sample = sample_padded
    
    # crop
    crop_size = np.random.randint(crop_minlen, min(input_size, sample.shape[1]) + 1)
    start_idx = np.random.randint(0, sample.shape[1] - crop_size + 1)
    sample = sample[:, start_idx : start_idx + crop_size]

    # unified shape
    sample_padded = np.zeros((2, input_size), dtype=np.float32)
    sample_padded[:, : sample.shape[1]] = sample
    sample = sample_padded
    return sample


NPY_EXTENSIONS = ".npy"


class MyFolder(DatasetFolder):
    def __init__(self, root, mode, input_size, dataset_idx=0, domain_classnum=0):
        super().__init__(
            root=root,
            loader=np.load,
            extensions=NPY_EXTENSIONS,
        )
        self.mode = mode
        self.input_size = input_size
        self.dataset_idx = dataset_idx
        self.domain_classnum = domain_classnum

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]

        sample = self.loader(path)
        if self.mode == "train":
            sample = random_crop_resize(sample, 512, self.input_size)
        else:
            sample = random_crop_resize(sample, min(len(sample), self.input_size), self.input_size)
        sample = torch.from_numpy(sample)
        target = self.dataset_idx * self.domain_classnum + target

        return sample, target


def get_dataset(dataset, domains, input_size):
    dir_dict = {
        "mail": "signal_pretrain",
    }
    root_dir = f"~/data/signal/{dir_dict[dataset]}"
    train_dset = []
    test_dset = []
    for domain in domains:
        train_dir = os.path.join(root_dir, domain, "train")
        test_dir = os.path.join(root_dir, domain, "val")

        train_dset.append(MyFolder(train_dir, "train", input_size))
        test_dset.append(MyFolder(test_dir, "test", input_size))

    return ConcatDataset(train_dset), ConcatDataset(test_dset)
