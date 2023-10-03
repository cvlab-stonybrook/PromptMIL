import os.path

import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm
import kornia as K
try:
    import jpeg4py as jpeg
    use_jpeg4py = True
except:
    use_jpeg4py = False


def read_rgb_img(img_p):
    if use_jpeg4py and img_p.lower().endswith((".jpg", "jpeg")):
        img = jpeg.JPEG(img_p).decode()
    else:
        img = cv2.cvtColor(cv2.imread(img_p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img


class MergePatchWsiDataset(Dataset):

    def __init__(self, dataset_root, dataset_csv_path, data_type, data_ext=".jpg", classes_names=None,
                 drop_out=0., val_fold_id=-1, **kwargs):
        super().__init__()
        if classes_names is None:
            self.CLASSES = None
            self.CLASS_NAMES = None
        else:
            self.CLASSES = classes_names[0]
            self.CLASS_NAMES = classes_names[1]

        self.dataset_root = dataset_root
        self.dataset_csv_path = dataset_csv_path
        self.data_ext = data_ext

        if data_type not in ['train', 'validation', 'test']:
            raise Exception("Not supported dataset type. It should be train or test")
        self.data_type = data_type
        self.val_fold_id = val_fold_id
        if data_type == 'test':
            self.val_fold_id = -1

        if val_fold_id >= 0:
            self.wsi_list = self.read_cv_dataset_csv()
        else:
            if data_type == 'validation':
                data_type = 'test'
                self.data_type = data_type
            self.wsi_list = self.read_dataset_csv()

        self.drop_out = drop_out

    def read_dataset_csv(self):
        df = pd.read_csv(self.dataset_csv_path, header=0)
        if self.data_type in ['test']:
            df = df[df['is_test'] > 0]
        else:  # train
            df = df[df['is_test'] == 0]
        return df

    def read_cv_dataset_csv(self):
        df = pd.read_csv(self.dataset_csv_path, header=0)
        if self.data_type in ['validation']:
            df = df[df['fold'] == self.val_fold_id]
        elif self.data_type in ['test']:
            df = df[df['fold'] < 0]
        else:
            df = df[df['fold'] > 0]
            df = df[df['fold'] != self.val_fold_id]
        return df

    def __len__(self):
        return len(self.wsi_list)

    def __getitem__(self, i):
        row = self.wsi_list.iloc[i]
        wsi_id = row['wsi_id']
        label = row['label']
        len_img = row['len_img']

        tiles = []
        for i in range(len_img):
            tile = read_rgb_img(os.path.join(self.dataset_root, "%s_%d%s" % (wsi_id, i, self.data_ext)))
            assert len(tile.shape) == 3
            h, w, c = tile.shape
            tile = tile.reshape(h // w, w, w, c)
            tiles.append(tile)
        tiles = np.concatenate(tiles, axis=0)

        tiles = K.utils.image_to_tensor(tiles)
        tiles = K.enhance.normalize(tiles, torch.tensor(0.), torch.tensor(255.))

        if self.drop_out > 0:
            perm = torch.randperm(tiles.size(0))
            idx = perm[:int((1 - self.drop_out) * tiles.size(0))]
            tiles = tiles[idx]

        # tiles = [t.half() for t in tiles]
        return tiles, label

    def get_weights_of_class(self):
        labels = self.wsi_list['label']
        unique, counts = np.unique(labels, return_counts=True)
        label_cnt = list(zip(unique, counts))
        label_cnt.sort(key=lambda x: x[0])
        weight_arr = np.array([x[1] for x in label_cnt], dtype=float)
        weight_arr = np.max(weight_arr) / weight_arr
        return torch.from_numpy(weight_arr.astype(np.float32))


class PatchWsiDataModule(pl.LightningDataModule):
    def __init__(self, dataset_root, dataset_csv, val_fold=-1, data_ext=".jpg", classes_names=None, num_workers=2,
                 drop_out=0., shuffule_train=True):
        super().__init__()
        if classes_names is None:
            self.CLASSES = None
            self.CLASS_NAMES = None
        else:
            self.CLASSES = classes_names[0]
            self.CLASS_NAMES = classes_names[1]

        self.dataset_root = dataset_root
        self.dataset_csv = dataset_csv
        self.data_ext = data_ext

        self.val_fold = val_fold

        self.num_workers = num_workers

        self.drop_out = drop_out

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

        self.shuffule_train = shuffule_train

    def setup(self, stage=None):
        if self.dataset_train is None:
            self.dataset_train = MergePatchWsiDataset(self.dataset_root, self.dataset_csv, "train",
                                                      data_ext=self.data_ext, val_fold_id=self.val_fold,
                                                      classes_names=[self.CLASSES, self.CLASS_NAMES],
                                                      drop_out=self.drop_out)
            self.dataset_val = MergePatchWsiDataset(self.dataset_root, self.dataset_csv, "validation",
                                                    data_ext=self.data_ext, val_fold_id=self.val_fold,
                                                    classes_names=[self.CLASSES, self.CLASS_NAMES],
                                                    drop_out=0.)
            self.dataset_test = MergePatchWsiDataset(self.dataset_root, self.dataset_csv, "test",
                                                     data_ext=self.data_ext, val_fold_id=self.val_fold,
                                                     classes_names=[self.CLASSES, self.CLASS_NAMES],
                                                     drop_out=0.)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=1, shuffle=self.shuffule_train, num_workers=self.num_workers,
                          drop_last=False, pin_memory=False)

    def val_dataloader(self):
        if self.val_fold >= 0:
            return (
                DataLoader(self.dataset_val, batch_size=1, shuffle=False, num_workers=self.num_workers,
                              drop_last=False, pin_memory=False),
                DataLoader(self.dataset_test, batch_size=1, shuffle=False, num_workers=self.num_workers,
                           drop_last=False, pin_memory=False),
            )
        else:
            return DataLoader(self.dataset_val, batch_size=1, shuffle=False, num_workers=self.num_workers,
                              drop_last=False, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=1, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=False)


if __name__ == '__main__':
    dataset_root = r'/data07/shared/jzhang/data/WSI/BRIGHT/patches/kapse_10x/merged_imgs'
    dataset_csv = r'/data07/shared/jzhang/data/WSI/BRIGHT/patches/kapse_10x/label_cv.csv'
    # dataset_root = r'/dev/shm/jzhang/TCGA-LUADSC/patches_kapse/5x'
    data_module = PatchWsiDataModule(dataset_root, dataset_csv, val_fold=0, classes_names=None, num_workers=4, drop_out=0.0)
    data_module.setup()

    print(len(data_module.dataset_train), len(data_module.dataset_val), len(data_module.dataset_test))

    wsi = data_module.dataset_train[0]
    print(wsi[0].shape, wsi[0].max(), wsi[0].min(), wsi[1])
    # print(wsi[0][0, 0, ...])

    print(data_module.dataset_train.get_weights_of_class())