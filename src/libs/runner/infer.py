import os.path as osp
import pandas as pd

from .manager import BaseManager
from utils import seed_everything, make_datapath_list, show
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

    def infer_oneimage(self, image):
        # PIL image (W, H)
        import numpy as np
        import cv2
        from dataset import OneTestDataset

        image = np.array(image, dtype=np.uint8)[:, :, ::-1] # to numpy
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # to cv2

        test_dataset = OneTestDataset(
            image, 
            "val", 
            DataTransform(self.image_size, self.get("val_transform_params"))
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
            show(image, preds, self.voc_classes, self.get("data_confidence_level"))



        return image
