import copy
import datetime
import json
import math
import os
import random
import time

import numpy as np
import torch
import utils.misc as misc
from methods import factory
from torch.utils.tensorboard import SummaryWriter
from utils.data_manager import DataManager


def train(args):
    log_dir = args.log_dir
    result_dir = args.result_dir
    seed_list = copy.deepcopy(args.seeds)

    for seed in seed_list:
        args.seed = seed
        args.log_dir = f"{log_dir}/seed{seed}"
        args.result_dir = f"{result_dir}/seed{seed}"

        print(f"seed: {args.seed}")
        print(f"log_dir: {args.log_dir}")
        print(f"result_dir: {args.result_dir}")

        _train(args)


def _train(args):
    set_random(args.seed)
    log_writer = set_exp(args)

    data_manager = DataManager(
        args.dataset,
        args.seed,
        args.limit_inst,
        args.init_inst,
        args.inc_inst,
        args.init_cls,
        args.inc_cls,
        args.domains.split(","),
    )
    args.domain_classnum = data_manager.domain_classnum
    model = factory.get_model(args.method, args)

    start_time = time.time()


    for task in range(data_manager.nb_tasks):
        model.before_task(data_manager)
        model.incremental_learn(task, data_manager, log_writer)
        model.after_task(data_manager)
        if args.checkpoint_dir and misc.is_main_process():
            model.save_checkpoint(args.checkpoint_dir)
        
    end_time = time.time()
    cost_time = end_time - start_time
    cost_time_str = str(datetime.timedelta(seconds=int(cost_time)))
    print(f"Cost time: {cost_time_str}\n")


def set_exp(args):
    eff_batch_size = args.train_batch_size * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"base lr: {(args.lr * 256 / eff_batch_size):.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"effective batch size: {eff_batch_size:d}")

    tuned_session = 1
    if args.inc_inst != 0:
        tuned_session *= (
            math.ceil((args.limit_inst - args.init_inst) / args.inc_inst) + 1
        )
    if args.tuned_epoch is None:
        args.tuned_epoch = args.base_epoch // tuned_session
    if args.method == "finetune":
        tuned_session *= (
            math.ceil((args.domain_classnum - args.init_cls) / args.inc_cls) + 1
        )

    print(f"base epoch: {args.base_epoch}")
    print(f"tuned epoch: {args.tuned_epoch} * {tuned_session}")

    if misc.is_main_process():
        dirs = [args.log_dir, args.result_dir]
        if args.checkpoint_dir:
            dirs.append(args.checkpoint_dir)

        for dir in dirs:
            os.makedirs(dir, exist_ok=True)
            config_filepath = os.path.join(dir, "configs.json")
            with open(config_filepath, "w") as fd:
                json.dump(
                    vars(args), fd, indent=2, sort_keys=True, cls=misc.ConfigEncoder
                )
        result_types = ["train", "test"]
        for result_type in result_types:
            results_filepath = os.path.join(
                args.result_dir, f"results_{result_type}.csv"
            )
            with open(results_filepath, "w", encoding="utf-8") as f:
                if os.path.getsize(results_filepath) != 0:
                    f.truncate(0)

        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    return log_writer


def set_random(seed):
    seed = seed + misc.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
