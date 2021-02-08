import os.path as osp
import pandas as pd

from .manager import BaseManager
from utils import seed_everything, make_datapath_list
from dataset import TestDataset, Anno_xml2list, DataTransform, od_collate_fn, get_dataloader
from models import ObjectDetectionModel

class Infer(BaseManager):
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
            testloader = get_dataloader(
                test_dataset, 
                batch_size=self.get("batch_size"), 
                num_workers=self.get("num_workers"), 
                shuffle=False, 
                drop_last=False
            )

            for seed in self.seeds:
                self.params["seed"] = seed
                self.params["phase"] = "inference" 
                
                model = ObjectDetectionModel(self.params)
                model.read_weight()
                preds = model.predict(testloader)
                try:
                    print(type(preds))
                    print(preds.shape)
                except:
                    pass
