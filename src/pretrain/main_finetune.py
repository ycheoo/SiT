import os
import argparse

from torch.distributed import destroy_process_group
from trainer_finetune import train
from utils.misc import load_json, init_distributed_mode

import setproctitle


def main():
    args = setup_parser().parse_args()
    config_param = load_json(
        os.path.join(
            "./configs", "pretrain", "finetune", args.dataset, f"{args.config}.json"
        )
    )
    args = vars(args)
    args.update(config_param)
    args = argparse.Namespace(**args)

    setproctitle.setproctitle(
        "{}_{}_{}".format("PRETRAIN_FINETUNE", args.dataset, args.config)
    )
    args.log_dir = f"./logs/pretrain/finetune/{args.dataset}/{args.config}/{args.seed}"
    args.output_dir = (
        f"./checkpoints/pretrain/finetune/{args.dataset}/{args.config}/{args.seed}"
    )
    checkpoint_mae = args.finetune
    args.finetune = f"./checkpoints/pretrain/{checkpoint_mae}"

    init_distributed_mode(args)
    train(args)
    destroy_process_group()


def setup_parser():
    parser = argparse.ArgumentParser(
        "MAE fine-tuning for signal classification", add_help=False
    )
    parser.add_argument("--dataset", default="mail", type=str, help="Dataset")
    parser.add_argument(
        "--config", default="default", type=str, help="Experiment config"
    )
    parser.add_argument("--domains", default="0531", type=str, help="Signal domains")
    parser.add_argument("--shots", default=-1, type=int, help="Few shot")
    parser.add_argument("--threshold", default=-1, type=int, help="Balance train dataset")

    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="sit_base",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument(
        "--input_size", default=224*224*3, type=int, help="images input size"
    )

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
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
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )

    # Augmentation parameters
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")

    # Dataset parameters
    parser.add_argument(
        "--nb_classes", default=12, type=int, help="number of the classification types"
    )

    parser.add_argument(
        "--output_dir",
        default="./logs/finetune",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./logs/finetune", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.set_defaults(dist_eval=True)
    
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
