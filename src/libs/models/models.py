from .base_model import BaseModel
from .networks import *
from utils import seed_everything
from utils_torch import MultiBoxLoss, run_training, inference_fn

import os.path as osp
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau



class BaseObjectDetectionModel(BaseModel):
    def fit(self, trainloader, validloader):

        optimizer = eval(self.optimizer)(self.model.parameters(), **self.params["optimizer_params"])
        scheduler = eval(self.scheduler)(optimizer, **self.params["scheduler_params"])
        criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=self.device)

        self.val_preds = run_training(
            model=self.model,
            trainloader=trainloader,
            validloader=validloader,
            epochs=self.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=criterion,
            early_stopping_steps=self.early_stopping_steps,
            verbose=self.verbose,
            device=self.device,
            seed=self.seed,
            weight_path=self.weight_path
        )

    def predict(self, testloader):
        preds = inference_fn(self.model, testloader, self.device)
        return preds
    def read_weight(self):
        fname = f"seed_{self.seed}.pt"
        self.model.load_state_dict(torch.load( osp.join(self.weight_path, fname) ))

    def save_weight(self):
        pass
    
    def perse_params(self):
        self.ROOT = self.params["ROOT"]
        self.WORK_DIR = self.params["WORK_DIR"]
        self.weight_path = osp.join(self.WORK_DIR, "weight")

        self.device = self.params["device"]
        self.epochs = self.params["epochs"]
        self.early_stopping_steps = self.params["early_stopping_steps"]
        self.verbose = self.params["verbose"]
        self.seed = self.params["seed"]
        self.phase = self.params["phase"]


        self.optimizer = self.params["optimizer"]
        self.scheduler = self.params["scheduler"]
    


class SimpleSSD(BaseObjectDetectionModel):
    def get_model(self):
        model = SSD(self.phase, self.params["model_cfg"])
        model.to(self.device)
        return model