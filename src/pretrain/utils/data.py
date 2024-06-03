import os
import random
from collections import defaultdict

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
        start_idx = np.random.randint(0, crop_minlen - sample.shape[1] + 1)
        sample_padded[:, start_idx : start_idx + sample.shape[1]] = sample
        sample = sample_padded

    # crop
    crop_size = np.random.randint(crop_minlen, min(input_size, sample.shape[1]) + 1)
    start_idx = np.random.randint(0, sample.shape[1] - crop_size + 1)
    sample = sample[:, start_idx : start_idx + crop_size]

    # unified shape
    sample_padded = np.zeros((2, input_size), dtype=np.float32)
    start_idx = np.random.randint(0, input_size - sample.shape[1] + 1)
    sample_padded[:, start_idx : start_idx + sample.shape[1]] = sample
    sample = sample_padded
    return sample


NPY_EXTENSIONS = ".npy"

ground_classes = ["210", "211", "212", "213", "214", "215", "216", "217"]
# ground_classes.sort()
# ground_class_to_idx = {cls_name: i for i, cls_name in enumerate(ground_classes)}

class MyFolder(DatasetFolder):
    def __init__(self, root, mode, input_size, is_mae=False, threshold=-1):
        super().__init__(
            root=root,
            loader=np.load,
            extensions=NPY_EXTENSIONS,
        )
        self.mode = mode
        self.is_mae = is_mae
        self.threshold = threshold
        self.input_size = input_size
        self.ground_classes = list(filter(lambda x: x in self.classes, ground_classes))
        self.ground_classes.sort()
        self.ground_class_to_idx = {cls_name: i for i, cls_name in enumerate(self.ground_classes)}
        
        if self.mode == "train" and self.threshold != -1:
            self.samples = self.filter_samples()
            
    def filter_samples(self):
        split_dict = defaultdict(list)
        
        for item in self.samples:
            path, target = item
            split_dict[target].append(item)

        filtered_dict = {}
        for key, value_list in split_dict.items():
            if len(value_list) > self.threshold:
                filtered_dict[key] = random.sample(value_list, self.threshold)
            else:
                filtered_dict[key] = value_list
        
        merged_list = []
        for value_list in filtered_dict.values():
            merged_list.extend(value_list)
        
        return merged_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]

        sample = self.loader(path)
        # For MAE, do not crop
        if self.mode == "train" and not self.is_mae:
            sample = random_crop_resize(sample, 512, self.input_size)
        else:
            sample = random_crop_resize(
                sample, min(len(sample), self.input_size), self.input_size
            )
        sample = torch.from_numpy(sample)
        # Specify class id
        target = self.ground_class_to_idx[self.classes[target]]

        return sample, target


def get_dataset(dataset, domains, input_size, is_mae=False, threshold=-1):
    dir_dict = {
        "mail": "signal_pretrain",
        "radar": "radar",
    }
    root_dir = f"~/data/{dir_dict[dataset]}"
    train_dset = []
    test_dset = []
    for domain in domains:
        train_dir = os.path.join(root_dir, domain, "train")
        test_dir = os.path.join(root_dir, domain, "val")

        train_dset.append(MyFolder(train_dir, "train", input_size, is_mae, threshold))
        test_dset.append(MyFolder(test_dir, "test", input_size))

    return ConcatDataset(train_dset), ConcatDataset(test_dset)
