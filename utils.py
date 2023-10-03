import json
import math
import os
import shutil
import sys
import random

import cv2
import torch
from albumentations import DualTransform
import albumentations.augmentations.crops.functional as F
from torch import nn


def save_parameters(args):
    folder_path = os.path.join(args.output_dir, args.run_name)
    os.makedirs(folder_path, exist_ok=True)
    args_dict = vars(args)
    with open(os.path.join(folder_path, "parameters.json"), "w") as f:
        json.dump(
            {n: str(args_dict[n]) for n in args_dict},
            f,
            indent=4
        )


def save_model_structure(args, model):
    folder_path = os.path.join(args.output_dir, args.run_name)
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, "model.txt"), "w") as f:
        f.write(str(model))

    # save the running source file
    source_file_path = sys.argv[0]
    source_file_name = os.path.split(source_file_path)[-1]
    try:
        shutil.copy2(source_file_path, os.path.join(folder_path, source_file_name))
    except:
        print("ERROR! Could not copy source file.")

class switch_dim(nn.Module):
    def forward(self, x):
        x = torch.transpose(x, 2, 1)
        return x