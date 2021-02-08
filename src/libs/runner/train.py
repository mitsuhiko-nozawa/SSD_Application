import sys, os
import os.path as osp
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .manager import BaseManager
from utils import seed_everything, make_datapath_list
#from utils_torch import get_transforms
from dataset import TrainDataset, Anno_xml2list, DataTransform, od_collate_fn
from models import *

class Train(BaseManager):
    def __init__(self, params):
        super(Train, self).__init__(params)
        self.device = torch.device(params["device"] if torch.cuda.is_available() else "cpu")
        self.model = params["model"]

        
    def __call__(self):
        # cvで画像を分ける
        print("Training")
        if self.get("train_flag"):
            for seed in self.seeds: # train by seed
                self.train_by_seed(seed)

    def train_by_seed(self, seed):
        seed_everything(seed)
        train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(self.data_path)
        if self.debug:
            train_img_list = train_img_list[:2]
            train_anno_list = train_anno_list[:2]
            val_img_list = val_img_list[:2]
            val_anno_list = val_anno_list[:2]

        train_dataset = TrainDataset(
            train_img_list, 
            train_anno_list, 
            phase="train", 
            transform=DataTransform(**self.get("tr_transform_params")),
            transform_anno=Anno_xml2list(self.voc_classes)
        )
        val_dataset = TrainDataset(
            val_img_list, 
            val_anno_list, 
            phase="val", 
            transform=DataTransform(**self.get("val_transform_params")),
            transform_anno=Anno_xml2list(self.voc_classes)
        )
        trainloader = DataLoader(
            train_dataset, 
            batch_size=self.get("batch_size"), 
            num_workers=self.get("num_workers"), 
            shuffle=True, 
            collate_fn=od_collate_fn,
        )
        validloader = DataLoader(
            val_dataset, 
            batch_size=self.get("batch_size"), 
            num_workers=self.get("num_workers"), 
            shuffle=False, 
            collate_fn=od_collate_fn,
        )
        self.params["seed"] = seed
        self.params["phase"] = "train" 
        
        model = eval(self.model)(self.params)
        model.fit(trainloader, validloader)
        # valid predict, no detect, 最後にdetectしてもいいかもね
        val_preds = model.val_preds
        print(val_preds[0].shape)
        print(val_preds[1].shape)
        print(val_preds[2].shape)
        #val_preds = pd.DataFrame(val_preds, columns=[f"pred_{n}" for n in range(self.params["output_size"])])
        #val_preds.to_csv(osp.join(self.val_preds_path, f"preds_{seed}_{fold}.csv"), index=False)

