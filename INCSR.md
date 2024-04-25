# SiT: Signal Transformer

## Requirements

```
pip install -r requirements.txt
```

## Configurations

1. **init_cls, inc_cls: initial class number and incremental class number.**
2. **init_inst, inc_inst, limit_inst: initial sample number, incremental sample number and limitation of samples used for training.**
3. **base_epoch: epoch for whole training stage (initial incremental training stage), and epoch for single training stage, tuned_epoch = base_epoch // tuned_session.**
4. **model_backbone_loc: path of pre-trained model to be loaded.**

## Experiments

```
./scripts/incsr/train_incsr.sh {node} {dataset} {config}
```

where `{dataset}` should be chosen from the name of subfolders from `configs/incsr` 
and `{config}` should be chosen from be selected from the json file in the specific subfolder.

## Datasets

### Organization of Datasets

`domain (day) - dtype (train or val) - categories (210, 211, etc) - sample file (.npy)`

```
├── 0531
│   ├── train
│   │   ├── 210
│   │   ├── 211
│   │   ├── 212
│   │   ├── 213
│   │   ├── 214
│   │   ├── 215
│   │   ├── 216
│   │   └── 217
│   └── val
│       ├── 210
│       ├── 211
│       ├── 212
│       ├── 213
│       ├── 214
│       ├── 215
│       ├── 216
│       └── 217
├── 0601
│   ├── train
│   │   ├── 210
│   │   ├── 211
│   │   ├── 212
│   │   ├── 213
│   │   ├── 214
│   │   ├── 215
│   │   ├── 216
│   │   └── 217
│   └── val
│       ├── 210
│       ├── 211
│       ├── 212
│       ├── 213
│       ├── 214
│       ├── 215
│       ├── 216
│       └── 217
```

### Formation of sample

1. **Sample could be both single or dual channels and any length.**
2. **Sample would be processed to the unified shape (2, 150528), with crop and padding.**
3. **Code for sample loader and processor is available at src/pretrain/utils/data.py.**

## Acknowledgments

This repo is based on [mae](https://github.com/facebookresearch/mae) and [PyCIL](https://github.com/G-U-N/PyCIL), many thanks.