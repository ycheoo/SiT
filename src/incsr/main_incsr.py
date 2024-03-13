import argparse
import json

from torch.distributed import destroy_process_group
from trainer_incsr import train
from utils.misc import init_distributed_mode

domains = "0531,0601,0602,0603,0605,0606,0607"


def main():
    args = get_args()
    init_distributed_mode(args)
    if args.config is not None:
        print(f"use json file config: {args.config}")
    train(args)
    destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser("Incremental signal recognition", add_help=False)

    parser.add_argument(
        "--config", type=str, default=None, help="Json file of settings."
    )

    parser.add_argument(
        "--method", default="incsr", type=str, help="Incremental learning method"
    )
    parser.add_argument("--model", default="incvpt", type=str, help="Model used")

    # Signal dataset parameters
    parser.add_argument(
        "--dataset", default="signal88", type=str, help="Signal dataset"
    )
    parser.add_argument("--domains", default="0531", type=str, help="Signal domains")

    # VPT parameters
    parser.add_argument("--vpt_type", default="deep", type=str, help="VPT type")
    parser.add_argument(
        "--prompt_token_num", default=5, type=int, help="Prompt token number"
    )

    # Train parameters
    parser.add_argument(
        "--train_batch_size",
        default=100,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--test_batch_size",
        default=100,
        type=int,
        help="Batch size per GPU (speed up evaluation under few-shot setting)",
    )
    parser.add_argument(
        "--base_epoch", type=int, default=200, metavar="N", help="epochs to tune"
    )
    parser.add_argument(
        "--tuned_epoch", type=int, default=None, metavar="N", help="epochs to tune"
    )

    # Incremental learning parameters
    parser.add_argument(
        "--init_inst", default=50, type=int, help="Number of initial class"
    )
    parser.add_argument(
        "--inc_inst", default=50, type=int, help="Number of incremental class"
    )
    parser.add_argument(
        "--limit_inst", default=200, type=int, help="Limit of instance per class"
    )
    parser.add_argument(
        "--init_cls", default=10, type=int, help="Number of initial class"
    )
    parser.add_argument(
        "--inc_cls", default=2, type=int, help="Number of incremental class"
    )

    # Model parameters
    parser.add_argument(
        "--model_name",
        default="vit_base_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--input_size", default=224 * 224 * 3, type=int, help="Unified input size"
    )

    # Optimizer parameters
    parser.add_argument(
        "--optimizer", type=str, default="sgd", help="optimizer (default sgd)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=0.1,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    # Model load and save params
    parser.add_argument("--model_backbone_loc", default="", help="Load from checkpoint")
    parser.add_argument("--model_pools_loc", default="", help="Load from checkpoint")
    parser.add_argument(
        "--checkpoint_dir",
        default="",
        help="path where to save checkpoint, empty for not saving",
    )

    # others
    parser.add_argument(
        "--log_dir", default="./logs/finetune", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--result_dir",
        default="./results/finetune",
        help="path where to save result",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seeds", default=[19899, 1872817, 914676, 0, 1], type=list)

    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    args = parser.parse_args()
    if args.config is not None:
        param = load_json(args.config)
        args = vars(args)
        args.update(param)
        args = argparse.Namespace(**args)
    return args


def load_json(setting_path):
    with open(setting_path) as f:
        param = json.load(f)
    return param


if __name__ == "__main__":
    main()
