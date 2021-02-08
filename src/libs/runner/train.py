import os.path as osp
import pandas as pd

from .manager import BaseManager
from utils import seed_everything, make_datapath_list
from dataset import TrainDataset, Anno_xml2list, DataTransform, od_collate_fn, get_dataloader
from models import ObjectDetectionModel

class Train(BaseManager):
    def __call__(self):
        print("Training")
        if self.get("train_flag"):
            for seed in self.seeds: 
                self.train_by_seed(seed)

    def train_by_seed(self, seed):
        seed_everything(seed)
        train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(self.data_path)
        if self.debug:
            train_img_list = train_img_list[:2]
            train_anno_list = train_anno_list[:2]
            val_img_list = val_img_list[:2]
            val_anno_list = val_anno_list[:2]
            self.params["epochs"] = 1

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
        trainloader = get_dataloader(
            dataset=train_dataset, 
            batch_size=self.get("batch_size"), 
            num_workers=self.get("num_workers"), 
            shuffle=True, 
            collate_fn=od_collate_fn,
            drop_last=False
        )
        validloader = get_dataloader(
            dataset=val_dataset, 
            batch_size=self.get("batch_size"), 
            num_workers=self.get("num_workers"), 
            shuffle=False, 
            collate_fn=od_collate_fn,
            drop_last=False
        )
        self.params["seed"] = seed
        self.params["phase"] = "train" 
        
        model = ObjectDetectionModel(self.params)
        model.fit(trainloader, validloader)
        # valid predict, no detect, 最後にdetectしてもいいかもね
        val_preds = model.val_preds
        try:
            print(val_preds[0].shape)
            print(val_preds[1].shape)
            print(val_preds[2].shape)
        except:
            pass
        #val_preds = pd.DataFrame(val_preds, columns=[f"pred_{n}" for n in range(self.params["output_size"])])
        #val_preds.to_csv(osp.join(self.val_preds_path, f"preds_{seed}_{fold}.csv"), index=False)

