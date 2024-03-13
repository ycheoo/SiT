import os
import argparse

from torch.distributed import destroy_process_group
from trainer_mae import train
from utils.misc import load_json, init_distributed_mode

import setproctitle


def main():
    args = setup_parser().parse_args()
    config_param = load_json(
        os.path.join(
            "./configs", "pretrain", "mae", args.dataset, f"{args.config}.json"
        )
    )
    args = vars(args)
    args.update(config_param)
    args = argparse.Namespace(**args)

    setproctitle.setproctitle(
        "{}_{}_{}".format("PRETRAIN_MAE", args.dataset, args.config)
    )
    args.log_dir = f"./logs/pretrain/mae/{args.dataset}/{args.config}/{args.seed}"
    args.output_dir = (
        f"./checkpoints/pretrain/mae/{args.dataset}/{args.config}/{args.seed}"
    )

    init_distributed_mode(args)
    train(args)
    destroy_process_group()


def setup_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument("--dataset", default="mail", type=str, help="Dataset")
    parser.add_argument(
        "--config", default="default", type=str, help="Experiment config"
    )
    parser.add_argument("--domains", default="0531", type=str, help="Signal domains")

    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="mae_vit_base_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument(
        "--input_size", default=224 * 224 * 3, type=int, help="images input size"
    )

    parser.add_argument(
        "--mask_ratio",
        default=0.75,
        type=float,
        help="Masking ratio (percentage of removed patches).",
    )

    parser.add_argument(
        "--norm_pix_loss",
        action="store_true",
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )
    parser.set_defaults(norm_pix_loss=True)

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
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
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )

    parser.add_argument(
        "--output_dir",
        default="./logs/imagenet",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./logs/imagenet", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    return parser


if __name__ == "__main__":
    main()
