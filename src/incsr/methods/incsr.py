import os
import random

import numpy as np
import utils.misc as misc
from engine_incsr import evaluate, extract_features, train_one_epoch
from methods.base import BaseLearner
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.data_manager import DataManager


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.features_list = np.empty((0, 1536))
        self.old_features_list = np.empty((0, 1536))
        self.labels_list = np.empty((0))
        self.old_labels_list = np.empty((0))

    def incremental_learn(self, task, data_manager: DataManager, log_writer):
        print(
            "Learning on {}-{} class, {} instance per class".format(
                self.known_classnum, self.total_classnum, self.total_instnum
            )
        )

        train_instrange = np.arange(0, self.total_instnum)
        train_classrange = np.arange(self.known_classnum, self.total_classnum)
        train_loader = self.gen_dataloader(
            data_manager,
            train_instrange,
            train_classrange,
            source="train",
            mode="train",
        )
        prototype_loader = self.gen_dataloader(
            data_manager, train_instrange, train_classrange, source="train", mode="test"
        )

        test_instrange = np.array([-1])
        test_classrange = np.arange(
            self.domain_classnum * self.known_domainnum, self.total_classnum
        )
        test_loader = self.gen_dataloader(
            data_manager,
            test_instrange,
            test_classrange,
            source="test",
            mode="test",
            shuffle=False,
            drop_last=False,
        )

        if self.newclass:
            if self.newdomain:
                self.network.switch_vpt(self.newlearn)
            self.network.update_fc(self.cur_domain_classnum)

        device = self.device
        self.network = DDP(
            self.network.to(device), device_ids=[device], find_unused_parameters=True
        )
        self.network_without_ddp = self.network.module

        if self.newdomain:
            self.train(train_loader, log_writer)
            self.network_without_ddp.switch_dualnetwork()

        self.replace_fc(prototype_loader)

        self.eval_curdomain(task, test_loader, log_writer)

        self.network = self.network_without_ddp

    def train(self, train_loader, log_writer):
        if self.optimizer == "sgd":
            optimizer = optim.SGD(
                self.network.parameters(),
                momentum=0.9,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            optimizer = optim.AdamW(
                self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.tuned_epoch, eta_min=self.min_lr
        )

        for epoch in range(self.tuned_epoch):
            train_loader.sampler.set_epoch(epoch)
            train_stats, outputs, targets = train_one_epoch(
                self.network,
                train_loader,
                optimizer,
                scheduler,
                self.device,
                epoch,
                self.domain_classnum,
            )
            cmt = confusion_matrix(targets, outputs)
            accs = []
            for i in range(cmt.shape[0]):
                acc = cmt[i, i] / np.sum(cmt[i, :])
                accs.append(acc)
            # make sure log_writer is not none
            if misc.is_main_process():
                log_writer.add_scalar(
                    "train/lr",
                    train_stats["lr"],
                    self.cur_task * self.tuned_epoch + epoch,
                )
                log_writer.add_scalar(
                    "train/train_loss",
                    train_stats["loss"],
                    self.cur_task * self.tuned_epoch + epoch,
                )
            results = {
                "step": self.cur_task * self.tuned_epoch + epoch,
                "task": self.cur_task,
                "epoch": epoch,
                "num_class": self.total_classnum,
                "num_instance": self.total_instnum,
                "loss": train_stats["loss"],
                "acc1": train_stats["acc1"],
                "acc5": train_stats["acc5"],
                "accs": "/".join(map(str, accs)),
            }
            misc.save_results("results_train", results, log_writer, self.result_dir)

    def replace_fc(self, data_loader):
        embed_list, target_list = extract_features(
            self.network, data_loader, self.device, self.domain_classnum
        )

        class_list = np.arange(
            self.known_classnum % self.domain_classnum, self.cur_domain_classnum
        )
        features_list = []
        labels_list = []
        for class_index in class_list:
            data_index = (target_list == class_index).nonzero().squeeze(-1)
            features = embed_list[data_index]
            targets = target_list[data_index]
            features_list.append(features.cpu().numpy())
            labels_list.append(targets.cpu().numpy())
            proto = features.mean(0)
            self.network_without_ddp.replace_fc(class_index, proto)
        features_list = np.vstack(features_list)
        labels_list = np.concatenate(labels_list)
        if self.newlearn:
            self.old_features_list = self.features_list
            self.old_labels_list = self.labels_list
        self.features_list = np.vstack((self.old_features_list, features_list))
        self.labels_list = np.concatenate((self.old_labels_list, labels_list))
        print(self.features_list.shape)
        print(self.labels_list.shape)
        features_list_path = os.path.join(
            self.result_dir, f"task_{self.cur_task}_features.npy"
        )
        np.save(features_list_path, self.features_list)
        labels_list_path = os.path.join(
            self.result_dir, f"task_{self.cur_task}_labels.npy"
        )
        np.save(labels_list_path, self.labels_list)

        protos = []
        for class_index in range(self.total_classnum):
            proto = self.network_without_ddp.fc.weight.data[class_index].detach()
            protos.append(proto.cpu().numpy())
        protos = np.vstack(protos)
        print(protos.shape)
        protos_path = os.path.join(self.result_dir, f"task_{self.cur_task}_protos.npy")
        np.save(protos_path, protos)

    def eval_curdomain(self, task, test_loader, log_writer):
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
