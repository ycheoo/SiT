import copy

import torch
from models.linears import CosineLinear
from models.vpt import VPT
from torch import nn


class IncVPT(nn.Module):
    def __init__(
        self,
        model_backbone_loc,
        vpt_type,
        prompt_token_num,
        depth=12,
        embed_dim=768,
        model_pools_loc=None,
    ):
        super().__init__()
        self.vpt = VPT(
            model_backbone_loc=model_backbone_loc,
            vpt_type=vpt_type,
            prompt_token_num=prompt_token_num,
            depth=depth,
            embed_dim=embed_dim,
        )
        self.vpt.freeze()

        self.depth = depth
        self.embed_dim = embed_dim
        self.feature_dim = embed_dim

        self.vpt_type = vpt_type
        self.prompt_token_num = prompt_token_num

        self.vptfc = None
        self.prompt_pool = nn.ParameterList()
        self.classifier_pool = nn.ModuleList()

        if model_pools_loc:
            pools = torch.load(model_pools_loc, map_location="cpu")
            self.prompt_pool = pools["prompt_pool"]
            self.classifier_pool = pools["classifier_pool"]
            self.set_dualmode()

    @property
    def dualmode(self):
        return self.feature_dim == 2 * self.embed_dim

    @property
    def fc(self):
        return self.classifier_pool[-1]

    @property
    def prompt(self):
        return self.prompt_pool[-1]

    def set_dualmode(self):
        self.feature_dim = 2 * self.embed_dim

    def generate_prompt(self):
        if self.vpt_type == "deep":
            prompt = nn.Parameter(
                torch.zeros(self.depth, self.prompt_token_num, self.embed_dim)
            )
        else:
            prompt = nn.Parameter(torch.zeros(1, self.prompt_token_num, self.embed_dim))
        return prompt

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def replace_fc(self, class_index, proto):
        self.fc.weight.data[class_index] = proto

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat(
                [weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()]
            )
            fc.weight = nn.Parameter(weight)
        self.classifier_pool[-1] = fc

    def forward(self, x):
        out = dict()
        if not self.dualmode:
            features = self.vpt(x, self.prompt)
        else:
            features = []
            features.append(self.extract_prototype(x))
            features.append(self.vpt(x, self.prompt))
            features = torch.cat(features, dim=1)
        out["logits"] = self.fc(features)
        out["features"] = features
        return out

    def extract_prototype(self, x):
        features = self.vpt.extract_prototype(x)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def switch_vpt(self, new_learn):
        print("switch to vpt")
        self.feature_dim = self.embed_dim

        if new_learn:
            self.vptfc = None
            if len(self.prompt_pool) != 0:
                self.prompt_pool[-1].require_grad = False
                self.classifier_pool[-1].require_grad = False
            prompt = self.generate_prompt()
            print("new prompt generated")
            self.prompt_pool.append(prompt)
            self.classifier_pool.append(None)
        if self.vptfc is not None:
            self.classifier_pool[-1] = self.vptfc
        else:
            self.classifier_pool[-1] = None

    def switch_dualnetwork(self):
        print("switch to dualnetwork")

        self.vptfc = copy.deepcopy(self.fc)

        self.set_dualmode()
        cur_classnum = self.fc.out_features
        fc = self.generate_fc(self.feature_dim, cur_classnum).cuda()
        self.classifier_pool[-1] = fc
