import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torchmetrics import MetricCollection, Accuracy, AUROC
from torchmetrics import F1Score as F1
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision import models

import network.get_network
from dataset import get_class_names
from dataset.merge_patch_wsi_dataset import PatchWsiDataModule
from options import get_arguments, get_arguments_additional
from pl_model.mil_e2e_trainer import MilE2EModule
from utils import save_parameters, switch_dim

import kornia.augmentation as K


def get_transforms(args):
    mean = args.data_mean if args.data_mean is not None else [0.485, 0.456, 0.406]
    std = args.data_std if args.data_std is not None else [0.229, 0.224, 0.225]
    if args.data_norm:
        transforms_train = nn.Sequential(
            # K.RandomResizedCrop((224, 224), scale=(0.4, 1.0), p=0.8),
            # K.RandomHorizontalFlip(),
            # K.ColorJitter(0.1, 0.1, 0.1, 0.1),
            # K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
            K.Normalize(mean=torch.tensor(mean),
                        std=torch.tensor(std))
        )
        transforms_eval = nn.Sequential(
            K.Normalize(mean=torch.tensor(mean),
                        std=torch.tensor(std))
        )
    else:
        transforms_train = nn.Sequential(
        )
        transforms_eval = nn.Sequential(
        )
    return transforms_train, transforms_eval


def get_metric(num_classes, task):
    metrics = MetricCollection({
        "Accuracy": Accuracy(num_classes=num_classes, task=task),
        "BA": Accuracy(num_classes=num_classes, average="macro", task=task),
        "F1": F1(num_classes=num_classes, task=task),
        "AUROC": AUROC(num_classes=num_classes, task=task),
    })
    return metrics


def get_network(args):
    if args.network.startswith("dino"):
        from network.get_network import get_dino_prompt_vit
        backbone = get_dino_prompt_vit(args.network, args.transfer_type, pretrained=args.load_backbone_weight,
                                      num_prompt_tokens=args.num_prompt_tokens,
                                      prompt_drop_out=args.prompt_dropout,
                                      project_prompt_dim=args.project_prompt_dim,
                                      deep_prompt=args.deep_prompt)
        num_fts = backbone.num_features
    elif args.network.startswith("hipt"):
        from network.get_network import get_hipt
        backbone = get_hipt(args.network, args.transfer_type, pretrained=args.load_backbone_weight,
                            num_prompt_tokens=args.num_prompt_tokens,
                            prompt_drop_out=args.prompt_dropout,
                            project_prompt_dim=args.project_prompt_dim,
                            deep_prompt=args.deep_prompt)
        num_fts = backbone.num_features
    elif args.network.startswith("transpath"):
        from network.get_network import get_prompt_transpath
        backbone = get_prompt_transpath(args.network, args.transfer_type, pretrained=args.load_backbone_weight,
                                          num_prompt_tokens=args.num_prompt_tokens,
                                          prompt_drop_out=args.prompt_dropout,
                                          project_prompt_dim=args.project_prompt_dim,
                                          deep_prompt=args.deep_prompt)
        num_fts = backbone.num_features
    else:
        from network.get_network import get_prompt_vit
        backbone = get_prompt_vit(args.network, args.transfer_type, pretrained=args.pretrained,
                                  num_prompt_tokens=args.num_prompt_tokens,
                                  prompt_drop_out=args.prompt_dropout,
                                  project_prompt_dim=args.project_prompt_dim,
                                  deep_prompt=args.deep_prompt)
        num_fts = backbone.num_features
    return backbone, num_fts


def get_loss_weight(args, data_module):
    if args.loss_weight is not None:
        loss_weight = args.loss_weight
    elif args.auto_loss_weight:
        data_module.setup()
        loss_weight = data_module.dataset_train.get_weights_of_class()
    else:
        loss_weight = None
    if loss_weight is not None:
        print("Using loss weight:", loss_weight)
        loss_weight = torch.Tensor(loss_weight)
    return loss_weight


def get_mil_network(mil_type, num_fts, num_classes, args, loss_weight=None):
    if mil_type in ["dsmil", "hipt_dsmil", "dsmil_bin", "dsmil_ce"]:
        from network.dsmil import FCLayer, BClassifier, MILNet
        i_classifier = FCLayer(in_size=num_fts, out_size=num_classes)
        b_classifier = BClassifier(input_size=num_fts, output_class=num_classes, dropout_v=args.dropout_att)
        classifier_model = MILNet(i_classifier, b_classifier)
        if mil_type in ["dsmil_ce"]:
            loss = nn.CrossEntropyLoss(weight=loss_weight)
        else:
            loss = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
    elif mil_type in ("clam_sb", "clam_mb"):
        from network.model_clam import CLAM_SB, CLAM_MB
        CLAM = CLAM_SB if mil_type == "clam_sb" else CLAM_MB
        clam_model_dict = {"dropout": True, 'n_classes': num_classes, 'subtyping': True, "size": args.clam_size,
                           'k_sample': 8, 'bag_weight': 0.7} #[192, 128, 128],
        classifier_model = CLAM(**clam_model_dict, instance_loss_fn='svm')
        loss = nn.CrossEntropyLoss(weight=loss_weight)
    elif mil_type in ["avgpooling", "maxpooling", "abmil", "gabmil"]:
        model_name = mil_type
        if model_name == "avgpooling":
            pooling_layer = nn.AdaptiveAvgPool1d(1)
        elif model_name == "maxpooling":
            pooling_layer = nn.AdaptiveMaxPool1d(1)
        elif model_name == "abmil":
            from network.pooling import AttentionPooling
            pooling_layer = AttentionPooling(num_fts, 128, out_dim=1, flatten=True, dropout=args.dropout_att)
        elif model_name == "gabmil":
            from network.pooling import GatedAttentionPooling
            pooling_layer = GatedAttentionPooling(num_fts, 128, out_dim=1, flatten=True, dropout=args.dropout_att)
        else:
            raise NotImplementedError

        classifier_model = nn.Sequential(
            switch_dim(),
            pooling_layer,
            nn.Flatten(),
            nn.Linear(num_fts, num_classes),
            # nn.Linear(512, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, num_classes)
        )
        loss = nn.CrossEntropyLoss(weight=loss_weight)
    else:
        raise NotImplementedError
    return classifier_model, loss


def get_model(args, backbone, num_fts, num_classes, loss_weight=None):
    from pl_model.forward_fn import model_to_classifier_type
    task = "multiclass"
    classifier_model, loss = get_mil_network(args.model, num_fts, num_classes, args, loss_weight=loss_weight)
    classifier_type = model_to_classifier_type[args.model]
    if args.model in ["hipt_hipt", "hipt_dsmil"]:
        trainer_model = HiptModule(backbone, classifier_model, loss, get_metric(num_classes, task),
                                   get_transforms(args), args, num_classes=num_classes,
                                   classifier_type=classifier_type)
    else:
        trainer_model = MilE2EModule(backbone, classifier_model, loss, get_metric(num_classes, task),
                                     get_transforms(args), args, num_classes=num_classes,
                                     classifier_type=classifier_type)
    return trainer_model


def main(args):
    classes_names = get_class_names(args.dataset_name)
    data_module = PatchWsiDataModule(args.dataset_root, args.dataset_csv, classes_names=classes_names,
                                     val_fold=args.val_fold, num_workers=args.num_workers, drop_out=args.dropout_inst)

    num_classes = len(classes_names[0])

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    backbone, num_fts = get_network(args)

    loss_weight = get_loss_weight(args, data_module)

    trainer_model = get_model(args, backbone, num_fts, num_classes, loss_weight)

    logger = WandbLogger(project=args.run_name, name=args.tag, log_model=False)

    trainer = pl.Trainer(default_root_dir=os.path.join(args.output_dir, args.run_name),
                         max_epochs=args.epochs, log_every_n_steps=50, num_sanity_val_steps=0,
                         precision=args.precision,
                         accelerator="gpu", devices=args.gpu_id,
                         logger=logger,
                         callbacks=[lr_monitor],
                         strategy='ddp' if len(args.gpu_id) > 1 else "auto",
                         )

    trainer.fit(trainer_model, data_module)
    
    if len(args.gpu_id) > 1:
        torch.distributed.destroy_process_group()
        if trainer.is_global_zero:
            trainer = pl.Trainer(default_root_dir=os.path.join(args.output_dir, args.run_name), 
                                 num_sanity_val_steps=0, logger=logger,
                                 accelerator="gpu", devices=[args.gpu_id[0]], )
            trainer.test(trainer_model, data_module)
    else:
        trainer.test(trainer_model, data_module)


def add_argument_fun(parser):
    parser.add_argument("--clam-size", type=lambda s: [int(item) for item in s.split(',')], default=[192, 128, 128],
                        help="Choose the number of samples")
    return parser


def process_argument_fun(opts):
    return opts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = get_arguments_additional(parser, add_argument_fun, process_argument_fun)

    save_parameters(args)

    main(args)
