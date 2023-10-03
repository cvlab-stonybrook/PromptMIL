from copy import deepcopy
from functools import partial

from torch import nn

def set_transfer_type(model, transfer_type):

    print("Using trasfer type %s" % transfer_type)
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
    elif transfer_type == "frozen" or transfer_type == "mil":
        for k, p in model.named_parameters():
            p.requires_grad = False
    elif transfer_type == "end2end":
        print("Enable all parameters update during training")
    else:
        raise NotImplementedError
    return model


def get_prompt_vit(variant, transfer_type, pretrained=False,
                   num_prompt_tokens=1,
                   prompt_drop_out=0.,
                   project_prompt_dim=-1,
                   deep_prompt=False):
    from .vit_prompt import PromptedTransformer, default_cfgs, update_pretrained_cfg_and_kwargs, load_pretrained, \
        load_custom_pretrained, checkpoint_filter_fn
    if variant == "vit_tiny_patch16_224":
        model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    elif variant == "vit_small_patch16_224":
        model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6,)
    else:
        raise NotImplementedError

    model = PromptedTransformer(model_kwargs, num_prompt_tokens, prompt_drop_out,
                                deep_prompt=deep_prompt, project_prompt_dim=project_prompt_dim)

    pretrained_cfg = deepcopy(default_cfgs[variant])

    update_pretrained_cfg_and_kwargs(pretrained_cfg, model_kwargs, None)
    pretrained_cfg.setdefault('architecture', variant)

    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat

    num_classes_pretrained = getattr(model, 'num_classes', model_kwargs.get('num_classes', 1000))

    pretrained_custom_load = 'npz' in pretrained_cfg['url']
    if pretrained:
        print("loading ImageNet pretrained weights.")
        if pretrained_custom_load:
            load_custom_pretrained(model, pretrained_cfg=pretrained_cfg)
        else:
            load_pretrained(
                model,
                pretrained_cfg=pretrained_cfg,
                num_classes=num_classes_pretrained,
                in_chans=model_kwargs.get('in_chans', 3),
                filter_fn=checkpoint_filter_fn,
                strict=False)

    model.head = nn.Identity()

    set_transfer_type(model, transfer_type)
    return model

def get_dino_prompt_vit(variant, transfer_type, pretrained=None, checkpoint_key="teacher",
                       num_prompt_tokens=1,
                       prompt_drop_out=0.,
                       project_prompt_dim=-1,
                       deep_prompt=False):
    from .dino_vit_prompt import DinoPromptedTransformer, dino_load_pretrained_weights
    if variant == "dino_vit_tiny_patch16_192":
        model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, use_avgpool=True, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    else:
        raise NotImplementedError

    model = DinoPromptedTransformer(model_kwargs, num_prompt_tokens, prompt_drop_out,
                                    deep_prompt=deep_prompt, project_prompt_dim=project_prompt_dim)


    if pretrained is not None:
        print("loading SSL pretrained weights.")
        dino_load_pretrained_weights(model, pretrained, checkpoint_key)
    else:
        print("Did not load weights!!!!!")

    set_transfer_type(model, transfer_type)
    return model


def get_hipt(variant, transfer_type, pretrained=None,
                       num_prompt_tokens=1,
                       prompt_drop_out=0.,
                       project_prompt_dim=-1,
                       deep_prompt=False):
    from .hipt.hipt_prompt import PromptHIPT4K

    if variant == "hipt":
        model = PromptHIPT4K(num_prompt_tokens, prompt_drop_out,
                             deep_prompt=deep_prompt, project_prompt_dim=project_prompt_dim)
    elif variant == "hipt_10x":
        model = PromptHIPT4K(num_prompt_tokens, prompt_drop_out,
                             deep_prompt=deep_prompt, project_prompt_dim=project_prompt_dim,
                             w_256=8, h_256=8)
    else:
        raise NotImplementedError

    if pretrained is not None:
        print("loading SSL pretrained weights.")
        model.load_weights(pretrained)
    else:
        print("Did not load weiths!!!!!")

    set_transfer_type(model, transfer_type)
    return model

def get_prompt_transpath(variant, transfer_type, pretrained=None,
                       num_prompt_tokens=1,
                       prompt_drop_out=0.,
                       project_prompt_dim=-1,
                       deep_prompt=False):
    from .transpath.transpath_prompt import PromptedTransPath
    from .transpath.modeling import CONFIGS
    if variant == "transpath":
        model_kwargs = CONFIGS['R50-ViT-B_16']
    else:
        raise NotImplementedError

    model = PromptedTransPath(model_kwargs, num_prompt_tokens, prompt_drop_out,
                              deep_prompt=deep_prompt, project_prompt_dim=project_prompt_dim)


    if pretrained is not None:
        print("loading SSL pretrained weights.")
        msg = model.load_state_dict(pretrained, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained, msg))
    else:
        print("Did not load weiths!!!!!")

    set_transfer_type(model, transfer_type)
    return model