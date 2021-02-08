import os
import os.path as osp
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(0)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    if torch.cuda.is_available(): 
        print("cuda available")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def make_datapath_list(rootpath):
    rootpath = Path(rootpath)

    voc2012_img_dir = Path(rootpath) / "JPEGImages"
    voc2012_anno_dir = Path(rootpath) / "Annotations"
    if not voc2012_img_dir.exists():
        raise Errors.FileNotFound(voc2012_img_dir)
    if not voc2012_anno_dir.exists():
        raise Errors.FileNotFound(voc2012_img_dir)

    id_names_root = rootpath / "ImageSets" / "Main"
    train_id_names = id_names_root / "train.txt"
    val_id_names = id_names_root / "val.txt"

    def glob_file_names_from_(id_names_file):
        img_list = []
        anno_list = []
        for line in open(id_names_file):
            file_id = line.strip() # 空白スペースと改行を除去
            img_list.append(str(voc2012_img_dir / f"{file_id}.jpg"))
            anno_list.append(str(voc2012_anno_dir / f"{file_id}.xml"))
        return img_list, anno_list
    ret = glob_file_names_from_(train_id_names)
    train_img_list = ret[0]
    train_anno_list = ret[1]
    ret = glob_file_names_from_(val_id_names)
    val_img_list = ret[0]
    val_anno_list = ret[1]

    return train_img_list, train_anno_list, val_img_list, val_anno_list
