import argparse
import os
import pickle
import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import jpeg4py as jpeg
from PIL import Image


def read_img_resize(img_p, patch_size):
    if img_p.lower().endswith((".jpg", "jpeg")):
        img = jpeg.JPEG(img_p).decode()
    else:
        img = cv2.cvtColor(cv2.imread(img_p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    if img.shape[0] != patch_size:
        img = resize_img(img, patch_size)
    return img

def resize_img(img, patch_size):
    return cv2.resize(img, (patch_size, patch_size))

def list_split(listA, n):
    for x in range(0, len(listA), n):
        every_chunk = listA[x: n+x]
        yield every_chunk

def process_batch(parallel, patch_paths, batch_size, patch_size):
    imgs = []
    for batch_paths in list_split(patch_paths, batch_size):
        patches = parallel(delayed(read_img_resize)(img_p=img_p, patch_size=patch_size) for img_p in batch_paths)
        # patches = [read_img_resize(img_p=img_p, patch_size=patch_size) for img_p in batch_paths]
        img = np.concatenate(patches, axis=0)
        imgs.append(img)
    return imgs


def main(args):
    parallel = Parallel(n_jobs=args.parallel_n, backend='loky')
    os.makedirs(args.output_dir, exist_ok=True)

    csv_rows = []

    for type in ["train", "test"]:
    # for type in ["test"]:
        prefix = args.train_prefix if type == "train" else args.test_prefix

        istest = '0' if type == "train" else '1'

        with open(os.path.join(args.in_dir, prefix + args.pickle_middle + args.list_suffix + ".pickle"), 'rb') as f:
            patch_list = pickle.load(f)
        with open(os.path.join(args.in_dir, prefix + args.pickle_middle + args.dict_suffix + ".pickle"), 'rb') as f:
            wsi_dic = pickle.load(f)

        num_patches = len(patch_list)
        num_wsi = len(wsi_dic)

        for wsi_id in tqdm(wsi_dic):
            if len(wsi_dic[wsi_id]) == 3:
                begin, end, label = wsi_dic[wsi_id]
            elif len(wsi_dic[wsi_id]) == 4:
                begin, end, label_1, label_2 = wsi_dic[wsi_id]
                label = label_1 * 2 + label_2
            patch_paths = patch_list[begin:end]

            out_imgs = process_batch(parallel, patch_paths, args.batch_size, args.patch_size)
            len_out_img = len(out_imgs)

            for i, out_img in enumerate(out_imgs):
                # print(out_img.shape, os.path.join(args.output_dir, "%s_%d%s" % (wsi_id, i, args.output_ext)))
                cv2.imwrite(os.path.join(args.output_dir, "%s_%d%s" % (wsi_id, i, args.output_ext)),
                            cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

            csv_rows.append({
                "wsi_id": wsi_id,
                "is_test": istest,
                "label": label,
                "len_img": len_out_img
            })
            # print(wsi_id, istest, label, len_out_img)

    data_frame = pd.DataFrame(data=csv_rows)
    data_frame.to_csv(args.output_csv_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("pickle_list_file", type=str, help='The path of input list pickle file.')
    # parser.add_argument("pickle_dict_file", type=str, help='The path of input dic pickle file.')
    parser.add_argument("in_dir", type=str, help='The path of the input list & dic directory')
    parser.add_argument("output_dir", type=str, help='The path of the output directory')
    parser.add_argument("output_csv_path", type=str, help='The path of the output csv file')

    parser.add_argument("--train_prefix", type=str, default="train_", help='')
    parser.add_argument("--test_prefix", type=str, default="test_", help='')
    parser.add_argument("--pickle_middle", type=str, default="5x_", help='')
    parser.add_argument("--list_suffix", type=str, default="list", help='')
    parser.add_argument("--dict_suffix", type=str, default="dict", help='')

    parser.add_argument("--output_ext", type=str, default=".jpg", help='output extension')
    parser.add_argument("--patch_size", type=int, default=224, help='The size of each patch')
    parser.add_argument("--batch_size", type=int, default=200, help='The size of each patch') # 200 is already the

    parser.add_argument("--parallel_n", type=int, default=10, help='The size of each patch')
    args = parser.parse_args()
    main(args)
    print("Done")
