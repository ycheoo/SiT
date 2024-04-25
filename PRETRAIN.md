# SiT: Signal Transformer

## Requirements

```
pip install -r requirements.txt
```

## Experiments

### Pretrain (mae and vit):

```
./scripts/pretrain/train_{episode}.sh {node} {dataset} {config}
```

`{episode}` should be be either mae or finetune, 
where `{dataset}` should be chosen from the name of subfolders from `configs/pretrain/{episode}` 
and `{config}` should be chosen from be selected from the json file in the specific subfolder.

### Training procedure

1. **Train mae, e.g., `./scripts/pretrain/train_mae.sh 2 mail default`.**
2. **Train vit, e.g., `./scripts/pretrain/train_finetune.sh 2 mail default`.**
3. **In the config file of vit (default.json), the finetune option should be set to specific path of pre-trained model, which can be both mae and vit.**

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