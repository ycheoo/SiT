import os

import numpy as np
import timm
import torch
import utils.misc as misc
from engine_incsr import evaluate
from models.inc_net import IncVPT
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, DistributedSampler
from utils.data_manager import DataManager


class BaseLearner(object):
    def __init__(self, args):
        self.device = args.gpu

        self.model = args.model
        if self.model == "incvpt":
            self.network = IncVPT(
                args.model_backbone_loc,
                args.vpt_type,
                args.prompt_token_num,
                model_pools_loc=args.model_pools_loc,
            )
        else:
            self.network = timm.create_model(
                self.model, pretrained=True, num_classes=args.domain_classnum
            )
            if args.model_backbone_loc != "":
                checkpoint = torch.load(args.model_backbone_loc, map_location="cpu")[
                    "model"
                ]
                self.network.load_state_dict(checkpoint)
        self.network_without_ddp = None

        self.cur_task = -1
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.lr = args.lr
        self.weight_decay = (
            args.weight_decay if args.weight_decay is not None else 0.0005
        )
        self.min_lr = args.min_lr if args.min_lr is not None else 1e-8
        self.num_workers = args.num_workers
        self.pin_mem = args.pin_mem
        self.optimizer = args.optimizer
        self.tuned_epoch = args.tuned_epoch

        self.known_domainnum = 0
        self.total_domainnum = 0

        self.known_classnum = 0
        self.total_classnum = 0
        self.domain_classnum = args.domain_classnum

        self.known_instnum = 0
        self.total_instnum = 0

        self.result_dir = args.result_dir

    @property
    def newlearn(self):
        return self.known_instnum == 0

    @property
    def newclass(self):
        return self.known_classnum != self.total_classnum

    @property
    def newdomain(self):
        return self.known_classnum % self.domain_classnum == 0

    @property
    def cur_domain_classnum(self):
        classnum = self.total_classnum % self.domain_classnum
        if classnum == 0 and self.total_classnum != 0:
            classnum = self.domain_classnum
        return classnum

    def save_checkpoint(self, checkpoint_dir):
        if self.model == "incvpt":
            pools = {
                "prompt_pool": self.network.prompt_pool,
                "classifier_pool": self.network.classifier_pool,
            }
            torch.save(pools, f"{checkpoint_dir}/pools.pth")
        else:
            raise NotImplementedError

    def before_task(self, data_manager: DataManager):
        self.cur_task += 1
        new_classnum, new_instnum = data_manager.get_task_size(self.cur_task)

        if new_classnum != 0:
            self.total_instnum = 0
            self.known_instnum = 0
            self.known_classnum = self.total_classnum
            if self.known_classnum % self.domain_classnum == 0:
                self.known_domainnum = self.total_domainnum

        self.total_instnum += new_instnum
        self.total_classnum += new_classnum
        self.total_domainnum = data_manager.get_domain_id(self.known_classnum) + 1

    def after_task(self, data_manager: DataManager):
        self.known_instnum = self.total_instnum
        if self.cur_task == data_manager.nb_tasks - 1:
            self.known_classnum = self.total_classnum
            self.known_domainnum = self.total_domainnum

    def gen_dataloader(
        self,
        data_manager: DataManager,
        inst_range,
        class_range,
        source,
        mode,
        shuffle=True,
        drop_last=True,
    ):
        rank = misc.get_rank()
        world_size = misc.get_world_size()
        batch_size = (
            self.train_batch_size if source == "train" else self.test_batch_size
        )
        dataset = data_manager.get_dataset(
            inst_range, class_range, source=source, mode=mode
        )
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=drop_last,
        )
        if source == "test" and len(dataset) % world_size != 0:
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )

        return data_loader

    def eval_curdomain(self, test_loader, log_writer):
        test_stats, outputs, targets = evaluate(
            self.network, test_loader, self.device, self.domain_classnum
        )
        cmt = confusion_matrix(targets, outputs)
        accs = []
        for i in range(cmt.shape[0]):
            acc = cmt[i, i] / np.sum(cmt[i, :])
            accs.append(acc)

        if misc.is_main_process():
            log_writer.add_scalar("test/test_loss", test_stats["loss"], self.cur_task)
            log_writer.add_scalar("test/test_acc1", test_stats["acc1"], self.cur_task)
            log_writer.add_scalar("test/test_acc5", test_stats["acc5"], self.cur_task)
            cmt_path = os.path.join(self.result_dir, f"task_{self.cur_task}_cmt.npy")
            np.save(cmt_path, cmt)

            results = {
                "task": self.cur_task,
                "num_class": self.total_classnum,
                "num_instance": self.total_instnum,
                "loss": test_stats["loss"],
                "acc1": test_stats["acc1"],
                "acc5": test_stats["acc5"],
                "accs": "/".join(map(str, accs)),
            }
            misc.save_results("results_test", results, log_writer, self.result_dir)

    def incremental_learn(self):
        pass
