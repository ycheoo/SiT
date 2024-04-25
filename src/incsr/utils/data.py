import os

import numpy as np
from torchvision import datasets, transforms


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    use_path = False
    class_order = None


def split_dataset(dataset):
    samples = []
    targets = []
    for item in dataset:
        samples.append(item[0])
        targets.append(item[1])

    return np.array(samples), np.array(targets)


def reduce_inst(seed, limit_inst, data, targets, class_order):
    np.random.seed(seed)
    reduced_data = []
    reduced_targets = np.repeat(class_order, repeats=limit_inst)
    for cls in class_order:
        indices = np.where(targets == cls)[0]
        subdata = data[indices]
        assert limit_inst <= len(subdata), "No enough instances."
        random_indices = np.random.choice(len(subdata), size=limit_inst, replace=False)
        reduced_data.append(subdata[random_indices])
    reduced_data = np.concatenate(reduced_data)
    return reduced_data, reduced_targets


def gen_domains(domain_id, dir_root):
    dataset = []
    classes = os.listdir(os.path.join(dir_root))
    classes.sort()
    classes = [
        "uav1",
        "uav2",
        "uav3",
        "uav4",
        "uav5",
        "uav6",
        "uav7",
        "uav8",
        "radio1",
        "radio2",
        "radio3",
        "radio4",
        "radio5",
        "radio6",
        "radio7",
        "radio8",
    ]
    print(classes)
    for cls_id, cls in enumerate(classes):
        samples = os.listdir(os.path.join(dir_root, cls))
        for sample_name in samples:
            sample = os.path.join(dir_root, cls, sample_name)
            target = cls_id + len(classes) * domain_id
            dataset.append((sample, target))
    return dataset


class iINCSR16(iData):
    def __init__(self):
        self.use_path = True
        self.domain_classnum = 16

    def download_data(self, seed, limit_inst, domains):
        root_dir = "/mnt/data/heyuchen/data/signal/signal_incsr"
        train_dataset = []
        test_dataset = []
        self.class_order = np.arange(self.domain_classnum * len(domains))

        for domain_id, domain in enumerate(domains):
            train_dir = os.path.join(root_dir, domain, "train")
            test_dir = os.path.join(root_dir, domain, "val")
            train_dataset.extend(gen_domains(domain_id, train_dir))
            test_dataset.extend(gen_domains(domain_id, test_dir))

        self.train_data, self.train_targets = split_dataset(train_dataset)
        self.test_data, self.test_targets = split_dataset(test_dataset)

        if not limit_inst == -1:
            self.train_data, self.train_targets = reduce_inst(
                seed, limit_inst, self.train_data, self.train_targets, self.class_order
            )
