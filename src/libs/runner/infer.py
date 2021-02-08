import sys, os
import os.path as osp
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .manager import BaseManager
from utils import seed_everything, make_datapath_list
#from utils_torch import get_transforms
from dataset import TestDataset, DataTransform
from models import *

class Infer(BaseManager):
    def __init__(self, params):
        super(Infer, self).__init__(params)
        self.device = torch.device(params["device"] if torch.cuda.is_available() else "cpu")
        self.model = params["model"]

    def __call__(self):
        print("Inference")
        if self.get("infer_flag"):
            _, __, val_img_list, val_anno_list = make_datapath_list(self.data_path)
            if self.debug:
                val_img_list = val_img_list[:2]
                val_anno_list = val_anno_list[:2]

            test_dataset = TestDataset(
                val_img_list, 
                phase="val", 
                transform=DataTransform(**self.get("val_transform_params")),
            )
            testloader = DataLoader(
                test_dataset, 
                batch_size=self.get("batch_size"), 
                num_workers=self.get("num_workers"), 
                shuffle=False, 
            )

            for seed in self.seeds:
                self.params["seed"] = seed
                self.params["phase"] = "inference" 
                
                model = eval(self.model)(self.params)
                model.read_weight()
                preds = model.predict(testloader)
                print(type(preds))
                print(preds.shape)
