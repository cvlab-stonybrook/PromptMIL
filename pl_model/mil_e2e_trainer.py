import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.optim import lr_scheduler

from .forward_fn import get_classifer_fuc


class MilE2EModule(pl.LightningModule):
    def __init__(self, backbone, classifier, loss, metrics, cus_transforms, args, num_classes,
                 classifier_type, **kwargs):
        super(MilE2EModule, self).__init__()
        # disable auto optim
        self.automatic_optimization = False

        self.args = args
        self.kwargs = kwargs

        self.backbone = backbone
        self.freeze_backbone = args.transfer_type in ["frozen", "mil"]
        self.classifier = classifier

        self.classifier_type = classifier_type
        self.classifer_fuc = get_classifer_fuc(classifier_type)

        self.loss = loss

        self.num_classes = num_classes

        self.lr = args.lr

        self.batch_size_train = args.batch_size_train
        self.batch_size_eval = args.batch_size_eval
        self.accumulate_grad_batches = args.accumulate_grad_batches

        self.train_metrics = metrics.clone(postfix='/train')
        self.valid_metrics = nn.ModuleList([metrics.clone(postfix='/val'), metrics.clone(postfix='/test')])
        self.test_metrics = metrics.clone(prefix='final_test/')

        if cus_transforms is None:
            self.transforms_train, self.transforms_eval = None, None
        elif isinstance(cus_transforms, (list, tuple)):
            self.transforms_train = cus_transforms[0]
            self.transforms_eval = cus_transforms[1]
        else:
            self.transforms_train, self.transforms_eval = cus_transforms, cus_transforms

        self.save_hyperparameters("args")

    def forward(self, data, label=None, train=False):
        # data, i = data
        # data = data.squeeze(0)
        ft = self.backbone_forward(data, train)
        if train:
            ft.requires_grad = True
        bag_prediction, loss, Y_prob = self.classifier_forward(ft, label)

        if train:
            self.manual_backward(loss)
            # loss.backward()
            if not self.freeze_backbone:
                self.backbone_backward(data, ft, train)
        return bag_prediction, loss, Y_prob

    def backbone_forward(self, data, train):
        fts = []
        with torch.no_grad():
            for data_i in self.split_tensor(data, self.batch_size_train):
                ft = self.backbone(data_i)
                fts.append(ft)
        fts = torch.cat(fts, dim=0)
        return fts

    def backbone_backward(self, data, fts, train):
        ft_grads = fts.grad

        data = self.split_tensor(data, self.batch_size_train)
        ft_grads = self.split_tensor(ft_grads, self.batch_size_train)

        for data_i, ft_grad_i in zip(data, ft_grads):
            new_ft_i = self.backbone(data_i)
            # self.manual_backward(new_ft_i, gradient=ft_grad_i)
            new_ft_i.backward(ft_grad_i)

    def classifier_forward(self, data, label=None):
        return self.classifer_fuc(data, self.classifier, self.loss, self.num_classes, label=label)

    def split_data(self, data, batch_size):
        # num_chk = int(np.ceil(data.shape[0] / batch_size))
        # return torch.chunk(data, num_chk, dim=0)
        if batch_size > 1:
            return [torch.cat(data[i:i + batch_size], dim=0) for i in range(0, len(data), batch_size)]
        else:
            return data

    def split_tensor(self, data, batch_size):
        num_chk = int(np.ceil(data.shape[0] / batch_size))
        return torch.chunk(data, num_chk, dim=0)

    def augmenat_data(self, data, train=False):
        data = data.squeeze(0)
        trans = self.transforms_train if train else self.transforms_eval
        if trans is not None:
            with torch.no_grad():
                for i in range(0, data.shape[0], self.batch_size_eval):
                    data[i:i+self.batch_size_eval, ...] = trans(data[i:i+self.batch_size_eval, ...])
        return data

    def training_step(self, batch, batch_idx):
        img, label = batch
        img = self.augmenat_data(img, train=True)

        y, loss, y_prob = self(img, label=label, train=True)

        opt = self.optimizers()
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt.step()
            opt.zero_grad()

        self.log("Loss/train", loss, on_step=True, on_epoch=True, sync_dist=True)
        #self.train_metrics.update(y_prob, label)
        self.train_metrics(y_prob, label)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        img, label = batch
        img = self.augmenat_data(img)

        y, loss, y_prob = self(img, label=label)

        if not self.trainer.sanity_checking:
            prefix = get_prefix_from_val_id(dataloader_idx)
            metrics_idx = dataloader_idx if dataloader_idx is not None else 0
            self.log("Loss/%s" % prefix, loss, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            #self.valid_metrics[metrics_idx].update(y_prob, label)
            self.valid_metrics[metrics_idx](y_prob, label)
            self.log_dict(self.valid_metrics[metrics_idx], on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        return loss

    def test_step(self, batch, batch_idx):
        img, label = batch
        img = self.augmenat_data(img)

        y, loss, y_prob = self(img, label=label)

        self.log("Loss/final_test", loss, on_step=False, on_epoch=True, sync_dist=True)
        #self.test_metrics.update(y_prob, label)
        self.test_metrics(y_prob, label)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, sync_dist=True)
        return self.test_metrics

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()


    def configure_optimizers(self):
        # if self.args.weight_decay is None:
        #     self.args.weight_decay = 1e-2
        # if self.args.delay_epoch > self.args.epochs:
        #     paras = [{"params": self.classifier.parameters()}]
        # else:
        paras = [{"params": filter(lambda p: p.requires_grad, self.backbone.parameters()), "lr": self.lr * self.args.lr_factor},
                 {"params": filter(lambda p: p.requires_grad, self.classifier.parameters()), }]
        if self.args.adam:
            cus_optimizer = torch.optim.Adam(paras, lr=self.lr, betas=(0.5, 0.9), weight_decay=self.args.weight_decay)
        else:
            cus_optimizer = torch.optim.AdamW(paras, lr=self.lr, weight_decay=self.args.weight_decay)
        cus_sch = torch.optim.lr_scheduler.CosineAnnealingLR(cus_optimizer, T_max=self.args.epochs, eta_min=5e-6)
        return {
            "optimizer": cus_optimizer,
            "lr_scheduler": cus_sch
        }


def get_prefix_from_val_id(dataloader_idx):
    if dataloader_idx is None or dataloader_idx == 0:
        return "val"
    elif dataloader_idx == 1:
        return "test"
    else:
        raise NotImplementedError
