import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iINCSR16


def select(x, y, idx):
    indices = np.where(y == idx)[0]
    return x[indices], y[indices]


def get_idata(dataset):
    name = dataset.lower()
    if name == "incsr16":
        return iINCSR16()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset))


class DataManager(object):
    def __init__(
        self,
        dataset,
        seed,
        limit_inst,
        init_inst,
        inc_inst,
        init_class,
        inc_class,
        domains,
    ):
        self.domains = domains
        self.dataset = dataset
        self.setup_data(dataset, seed, limit_inst, domains)

        self.inc_instlist = [init_inst]
        while sum(self.inc_instlist) + inc_inst < limit_inst:
            self.inc_instlist.append(inc_inst)
        offset_inst = limit_inst - sum(self.inc_instlist)
        if offset_inst > 0:
            self.inc_instlist.append(offset_inst)
        print(self.inc_instlist)

        assert init_class <= self.domain_classnum, "No enough classes."
        self.inc_classlist = [init_class]
        self.inc_classlist.extend([0] * (self.instlist_size - 1))
        while sum(self.inc_classlist) + inc_class < self.domain_classnum:
            self.inc_classlist.append(inc_class)
            self.inc_classlist.extend([0] * (self.instlist_size - 1))

        offset_class = self.domain_classnum - sum(self.inc_classlist)
        if offset_class > 0:
            self.inc_classlist.append(offset_class)
            self.inc_classlist.extend([0] * (self.instlist_size - 1))
        self.inc_classlist *= len(self.domains)
        print(self.inc_classlist)

    @property
    def instlist_size(self):
        return len(self.inc_instlist)

    @property
    def nb_tasks(self):
        return len(self.inc_classlist)

    def get_task_size(self, task):
        return self.inc_classlist[task], self.inc_instlist[task % self.instlist_size]

    def get_total_classnum(self):
        return len(self.class_order)

    def get_domain_classnum(self):
        return self.domain_classnum

    def get_domain_id(self, classnum):
        domain_id = classnum // self.domain_classnum
        return domain_id

    def setup_data(self, dataset, seed, limit_inst, domains):
        idata = get_idata(dataset)
        idata.download_data(seed, limit_inst, domains)

        self.train_data, self.train_targets = idata.train_data, idata.train_targets
        self.test_data, self.test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        self.domain_classnum = idata.domain_classnum
        self.class_order = np.unique(self.train_targets).tolist()
        print(self.class_order)

    def get_dataset(self, indices_inst, indices_class, source, mode):
        if source == "train":
            x, y = self.train_data, self.train_targets
        elif source == "test":
            x, y = self.test_data, self.test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in range(len(indices_class)):
            class_data, class_targets = select(x, y, indices_class[idx])
            if indices_inst[0] != -1:
                class_data = class_data[indices_inst]
                class_targets = class_targets[indices_inst]
            data.append(class_data)
            targets.append(class_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        return DummyDataset(self.dataset, mode, data, targets, trsf)


class DummyDataset(Dataset):
    def __init__(self, mode, dataset, samples, targets, trsf):
        assert len(samples) == len(targets), "Data size error!"
        self.mode = mode
        self.dataset = dataset
        self.samples = samples
        self.targets = targets
        self.trsf = trsf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        if not "cifar" in self.dataset:
            sample = np.load(path)
            crop_minlen = 512 if self.mode == "train" else min(len(sample), 224*224*3)
            sample = self.random_crop_resize(sample, crop_minlen, 224*224*3)
            sample = torch.from_numpy(sample)
        else:
            sample = self.trsf(Image.fromarray(path))

        target = self.targets[index]

        return sample, target

    def random_crop_resize(self, sample, crop_minlen, input_size):
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