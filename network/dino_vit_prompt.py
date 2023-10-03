from copy import deepcopy
from functools import reduce
from operator import mul
import math
import os
from functools import partial
import torch

from torch import nn
from torch.nn.modules.utils import _pair

from . import dino_vision_transformer as vit


class DinoPromptedTransformer(vit.VisionTransformer):
    """ Vision Transformer

        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929
        """

    def __init__(
            self,
            vit_config,
            num_tokens=1,
            drop_out=0.,
            project_prompt_dim=-1,
            deep_prompt=False,
            # freeze_backbone=True
    ):
        super().__init__(**vit_config)
        # if freeze_backbone:
        #     for param in self.parameters():
        #         param.requires_grad = False
        # self.prompt_config = prompt_config
        self.vit_config = vit_config

        self.num_prefix_tokens = 0 if self.use_avgpool else 1

        patch_size = _pair(vit_config["patch_size"])

        self.num_prompt_tokens = num_tokens  # number of prompted tokens
        self.deep_prompt = deep_prompt

        self.prompt_dropout = nn.Dropout(drop_out)

        # if project the prompt embeddings
        if project_prompt_dim > 0:
            # only for prepend / add
            prompt_dim = project_prompt_dim
            self.prompt_proj = nn.Linear(
                prompt_dim, vit_config["embed_dim"])
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = vit_config["embed_dim"]
            self.prompt_proj = nn.Identity()

        if num_tokens > 0:
            # initiate prompt:
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        else:
            pass

        if self.deep_prompt:  # noqa
            total_d_layer = vit_config["depth"] - 1
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                total_d_layer, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.prepare_tokens(x)

        prompt = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1))
        # if self.num_prefix_tokens > 0:
        x = torch.cat((
            x[:, :self.num_prefix_tokens, :],
            prompt,
            x[:, self.num_prefix_tokens:, :]
        ), dim=1)
        # else:
        #     x = torch.cat((prompt, x[:, :, :]), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        # x = self.norm_pre(x)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            if self.num_prompt_tokens > 0:
                super().train(False)
                self.prompt_proj.train()
                self.prompt_dropout.train()
            else:
                super().train(mode)
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output):
        # attn_weights = []
        hidden_states = None
        # weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config["depth"]

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.blocks[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i - 1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :self.num_prefix_tokens, :],
                        deep_prompt_emb,
                        hidden_states[:, (self.num_prefix_tokens + self.num_prompt_tokens):, :]
                    ), dim=1)

                hidden_states = self.blocks[i](hidden_states)

            # if self.encoder.vis:
            #     attn_weights.append(weights)

        # encoded = self.encoder.encoder_norm(hidden_states)
        # return encoded, attn_weights
        return hidden_states

    def forward_features(self, x):
        if self.num_prompt_tokens > 0:
            x = self.incorporate_prompt(x)
        else:
            x = self.prepare_tokens(x)

        if self.num_prompt_tokens > 0 and self.deep_prompt:
            x = self.forward_deep_prompt(x)
        else:
            for layer_count, blk in enumerate(self.blocks):
                x = blk(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):

        x = self.norm(x)  # B, L, C

        if not self.use_avgpool:
            return x[:, 0]
        else:
            x = x[:, self.num_prefix_tokens + self.num_prompt_tokens:].mean(dim=1)
            x = torch.flatten(x, 1)
            return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

def dino_load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if not os.path.isfile(pretrained_weights):
        print("wrong weight path")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

if __name__ == '__main__':

    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, use_avgpool=True, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))

    #
    model = DinoPromptedTransformer(model_kwargs, 1, 0., deep_prompt=False, project_prompt_dim=-1)

    lung_dino_path = "/data04/shared/skapse/Cell_guided/Experiments/Lung_cancer/DINO_5X/100_percent_data_ep100/vit_tiny_baseline_avgpool_fp16true_momentum996_outdim65536/checkpoint.pth"

    load_pretrained_weights(model, lung_dino_path, "teacher")

    transfer_type = "prompt"
    if transfer_type == "prompt":
        for k, p in model.named_parameters():
            if "prompt" not in k:
                p.requires_grad = False
    elif transfer_type == "cls":
        for k, p in model.named_parameters():
            if "cls_token" not in k:
                p.requires_grad = False
    elif transfer_type == "cls+prompt":
        for k, p in model.named_parameters():
            if "prompt" not in k and "cls_token" not in k:
                p.requires_grad = False
    elif transfer_type == "end2end":
        print("Enable all parameters update during training")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)  # , p.data)

    model = model.cuda()
    x = torch.rand(10, 3, 224, 224).cuda()
    y = model(x)
    print(y.shape)