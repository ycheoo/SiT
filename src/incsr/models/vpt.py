import timm
import torch
import torch.nn as nn
from timm.models.layers.trace_utils import _assert
from timm.models.vision_transformer import PatchEmbed, VisionTransformer


class PatchEmbed1D(nn.Module):
    """1D Signal to Patch Embedding"""

    def __init__(
        self,
        img_size=224*224*3,
        patch_size=16 * 16 * 3,
        in_chans=2,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.flatten = flatten
        self.num_patches = img_size // patch_size

        self.proj = nn.Conv1d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, SL = x.shape
        _assert(
            SL == self.img_size,
            f"Input sample length ({SL}) doesn't match model ({self.img_size}).",
        )
        x = self.proj(x)
        if self.flatten:
            x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class VPT(VisionTransformer):
    def __init__(
        self,
        img_size=224*224*3,
        patch_size=16 * 16 * 3,
        in_chans=2,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed1D,
        norm_layer=None,
        act_layer=None,
        model_backbone_loc=None,
        prompt_token_num=5,
        vpt_type="deep",
        use_head=False,
    ):
        # Recreate ViT
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        if model_backbone_loc:
            checkpoint = torch.load(model_backbone_loc, map_location="cpu")["model"]
        else:
            checkpoint = timm.create_model(
                "vit_base_patch16_224", pretrained=False, num_classes=0
            ).state_dict()
        self.load_state_dict(checkpoint, False)

        self.prompt_token_num = prompt_token_num
        self.vpt_type = vpt_type
        self.depth = depth
        if not use_head:
            self.head = None

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    with torch.no_grad():

        def extract_prototype(self, x):
            x = self.patch_embed(x)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = self.pos_drop(x + self.pos_embed)
            x = self.blocks(x)
            x = self.norm(x)
            x = x[:, 0, :]
            return x

    def forward(self, x, prompt_tokens):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x_token_num = x.shape[1]
        if prompt_tokens.dim() != 4:
            prompt_tokens = prompt_tokens.unsqueeze(0)
            prompt_tokens = prompt_tokens.expand(x.shape[0], -1, -1, -1)
        if self.vpt_type == "deep":
            for i in range(len(self.blocks)):
                prompt_token = prompt_tokens[:, i]
                x = torch.cat((x, prompt_token), dim=1)
                x = self.blocks[i](x)[:, :x_token_num]
        else:
            prompt_token = prompt_tokens[:, 0]
            x = torch.cat((x, prompt_token), dim=1)
            x = self.blocks(x)[:, :x_token_num]

        x = self.norm(x)
        x = x[:, 0, :]

        if self.head is not None:
            x = self.head(x)

        return x
