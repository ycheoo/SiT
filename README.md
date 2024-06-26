# SiT: Signal Transformer

## Requirements

```
pip install -r requirements.txt
```

## [Pretrain](./PRETRAIN.md)

## [Incsr](./INCSR.md)

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