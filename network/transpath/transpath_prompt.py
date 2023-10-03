import math
import torch
from functools import reduce
from operator import mul
from torch import nn
from torch.nn.modules.utils import _pair

from . import modeling

class PromptedTransPath(modeling.Transformer):
    def __init__(
            self,
            config,
            num_tokens=1,
            drop_out=0.,
            project_prompt_dim=-1,
            deep_prompt=False,

    ):
        super().__init__(config, 256, vis=False)
        self.vit_config = config

        patch_size = _pair(16)
        self.num_features = config.hidden_size

        self.num_prompt_tokens = num_tokens  # number of prompted tokens
        self.deep_prompt = deep_prompt

        self.prompt_dropout = nn.Dropout(drop_out)

        # if project the prompt embeddings
        if project_prompt_dim > 0:
            # only for prepend / add
            prompt_dim = project_prompt_dim
            self.prompt_proj = nn.Linear(
                prompt_dim, config.hidden_size)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = config.hidden_size
            self.prompt_proj = nn.Identity()

        if num_tokens > 0:
            # initiate prompt:
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        else:
            pass

        if self.deep_prompt and num_tokens > 0:  # noqa
            total_d_layer = config.transformer["num_layers"] - 1
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                total_d_layer, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

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

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                x[:, 1:, :]
            ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        return x

    def forward_deep_prompt(self, embedding_output):
        # attn_weights = []
        hidden_states = None
        # weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]

        attn_weights = []
        # for layer_block in self.layer:
        #     hidden_states, weights = layer_block(hidden_states)
        #     if self.vis:
        #         attn_weights.append(weights)
        # encoded = self.encoder_norm(hidden_states)
        # return encoded, attn_weights
        #
        for i in range(num_layers):
            if i == 0:
                hidden_states, weights = self.encoder.layer[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i - 1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1 + self.num_prompt_tokens):, :]
                    ), dim=1)

                    hidden_states, weights = self.encoder.layer[i](hidden_states)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward(self, x):
        if self.num_prompt_tokens > 0:
            x = self.incorporate_prompt(x)
        else:
            x = self.embeddings(x)

        if self.num_prompt_tokens > 0 and self.deep_prompt:
            x, attn_weights = self.forward_deep_prompt(x)
        else:
            x, attn_weights = self.encoder(x)
        x = x[:, 0]
        return x