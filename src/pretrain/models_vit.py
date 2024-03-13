# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer

from timm.models.layers.trace_utils import _assert

class PatchEmbed1D(nn.Module):
    """ 1D Signal to Patch Embedding
    """
    def __init__(self, img_size=224*224*3, patch_size=16*16*3, in_chans=1, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.flatten = flatten
        self.num_patches = img_size // patch_size

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, SL = x.shape
        _assert(SL == self.img_size, f"Input sample length ({SL}) doesn't match model ({self.img_size}).")
        x = self.proj(x)
        if self.flatten:
            x = x.transpose(1, 2)
        x = self.norm(x)
        return x

def sit_base(**kwargs):
    model = VisionTransformer(
        img_size=224*224*3, embed_layer=PatchEmbed1D, patch_size=16*16*3, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def sit_large(**kwargs):
    model = VisionTransformer(
        img_size=224*224*3, embed_layer=PatchEmbed1D, patch_size=16*16*3, in_chans=1, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
