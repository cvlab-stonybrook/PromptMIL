import argparse
import os
import time
from pprint import pprint

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from PIL import Image
import kornia as K
import jpeg4py as jpeg


saving_to_disk = False


def eval_transforms(pretrained=False):
    # if pretrained:
    #     mean = (0.485, 0.456, 0.406)
    #     std = (0.229, 0.224, 0.225)
    #     print("Using imagenet mean & std:", mean, std)
    # else:
    #     mean = (0.5, 0.5, 0.5)
    #     std = (0.1, 0.1, 0.1)
    #     print("Using uncondition mean & std:", mean, std)

    trnsfrms_val = transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Normalize(mean=mean, std=std)
        ]
    )
    return trnsfrms_val


class MergePatchWsiDataset(Dataset):

    def __init__(self, dataset_root, dataset_csv_path, data_ext=".jpg"):
        super().__init__()

        self.dataset_root = dataset_root
        self.dataset_csv_path = dataset_csv_path
        self.data_ext = data_ext
        self.wsi_list = self.read_dataset_csv()

    def read_dataset_csv(self):
        df = pd.read_csv(self.dataset_csv_path, header=0)
        return df

    def __len__(self):
        return len(self.wsi_list)

    def get_wsi_id(self, i):
        row = self.wsi_list.iloc[i]
        wsi_id = row['wsi_id']
        return wsi_id

    def __getitem__(self, i):
        row = self.wsi_list.iloc[i]
        wsi_id = row['wsi_id']
        label = row['label']
        len_img = row['len_img']

        tiles = []
        for j in range(len_img):
            tile = read_rgb_img(os.path.join(self.dataset_root, "%s_%d%s" % (wsi_id, j, self.data_ext)))
            assert len(tile.shape) == 3
            h, w, c = tile.shape
            tile = tile.reshape(h // w, w, w, c)
            tiles.append(tile)
        tiles = np.concatenate(tiles, axis=0)

        tiles = K.utils.image_to_tensor(tiles)
        tiles = K.enhance.normalize(tiles, torch.tensor(0.), torch.tensor(255.))

        # tiles = [t.half() for t in tiles]
        return tiles, i


def read_rgb_img(img_p):
    if img_p.lower().endswith((".jpg", "jpeg")):
        img = jpeg.JPEG(img_p).decode()
    else:
        img = cv2.cvtColor(cv2.imread(img_p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img


def get_network(args):
    if args.network.startswith("kapse"):
        from network.get_network import get_dino_prompt_vit
        backbone = get_dino_prompt_vit(args.network, "mil", pretrained=args.load_backbone_weight,
                                      num_prompt_tokens=0,
                                      prompt_drop_out=0.,
                                      deep_prompt=False)
        num_fts = backbone.num_features
    elif args.network.startswith("hipt"):
        from network.get_network import get_hipt
        backbone = get_hipt(args.network, "mil", pretrained=args.load_backbone_weight,
                            num_prompt_tokens=0, prompt_drop_out=0., deep_prompt=False)
        num_fts = backbone.num_features
    else:
        from network.get_network import get_prompt_vit
        backbone = get_prompt_vit(args.network, "mil", pretrained=args.pretrained,
                                  num_prompt_tokens=0, prompt_drop_out=0., deep_prompt=False)
        num_fts = backbone.num_features
    return backbone, num_fts


def split_tensor(data, batch_size):
    num_chk = int(np.ceil(data.shape[0] / batch_size))
    return torch.chunk(data, num_chk, dim=0)


def main(args):
    # dataset =
    os.makedirs(args.output_folder, exist_ok=True)
    ds = MergePatchWsiDataset(args.dataset_root, args.dataset_csv, data_ext=args.dataext)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # print(ds.get_wsi_id(0))

    # define network
    network, num_fts = get_network(args)
    # network.fc = nn.Identity()

    network = network.to(args.device)
    network.eval()

    # for batches
    for i, (wsi_patches, idx) in enumerate(tqdm(dl)):
        features = []
        wsi_id = ds.get_wsi_id(idx.numpy()[0])
        with torch.no_grad():
            wsi_patches = split_tensor(wsi_patches.squeeze(0), args.batch_size)
            for data_i in wsi_patches:
                data_i = data_i.to(args.device)
                ft_i = network(data_i)
                features.append(ft_i)
        features = torch.cat(features, dim=0).cpu()
        # print(idx, wsi_id, features.shape)
        torch.save(features, os.path.join(args.output_folder, "%s.pt" % wsi_id))
    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=str, help='Input folder containing all images')
    parser.add_argument("dataset_csv", type=str, help='the csv file')
    parser.add_argument("output_folder", type=str, help='Output folder')
    parser.add_argument("--dataext", type=str, default=".jpg", help="extension of images")
    parser.add_argument("--network", type=str, default=None, help="type of backbone network, eg. vit_tiny_patch16_384")
    parser.add_argument('--load-backbone-weight', type=str, default=None,
                        help='If not None, load weights from given path')
    parser.add_argument("--batch-size", type=int, default=512, help="Choose the batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="")
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    args.device = torch.device('cuda:%s' % args.gpu_id)
    main(args)
