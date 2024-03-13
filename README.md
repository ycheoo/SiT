# SiT: Signal Transformer

## Requirements

    pip install -r requirements.txt

## Experiments
Pretrain mae:

    ./scripts/pretrain/train_mae.sh --config {config}

where `{config}` should be chosen from `configs/pretrain/mae` folder, ablate `--config` for default setting.

Pretrain finetune:

    ./scripts/pretrain/train_finetune.sh --config {config}

where `{config}` should be chosen from `configs/pretrain/finetune` folder, ablate `--config` for default setting.

## Acknowledgments
This repo is based on [mae](https://github.com/facebookresearch/mae) and [PyCIL](https://github.com/G-U-N/PyCIL), many thanks.