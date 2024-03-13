import numpy as np
from methods.base import BaseLearner
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.data_manager import DataManager


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

    def incremental_learn(self, data_manager: DataManager, log_writer):
        print("Eval on {}-{} class".format(self.known_classnum, self.total_classnum))

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

        device = self.device
        self.network = DDP(
            self.network.to(device), device_ids=[device], find_unused_parameters=True
        )
        self.network_without_ddp = self.network.module

        self.eval_curdomain(test_loader, log_writer)

        self.network = self.network_without_ddp
