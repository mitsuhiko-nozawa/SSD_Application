from .base_model import BaseModel
from .networks import *
from .run_utils import run_training, inference_fn
from .criterion import MultiBoxLoss

import os.path as osp
import torch


class ObjectDetectionModel(BaseModel):
    def fit(self, trainloader, validloader):
        criterion = MultiBoxLoss(**self.params["criterion_params"], device=self.device)
        self.val_preds = run_training(
            model=self.model,
            trainloader=trainloader,
            validloader=validloader,
            epochs=self.epochs,
            optimizer=self.optimizer,
            optimizer_params=self.params["optimizer_params"],
            scheduler=self.scheduler,
            scheduler_params=self.params["scheduler_params"],
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

        self.device = torch.device(self.params["device"] if torch.cuda.is_available() else "cpu")
        self.epochs = self.params["epochs"]
        self.early_stopping_steps = self.params["early_stopping_steps"]
        self.verbose = self.params["verbose"]
        self.seed = self.params["seed"]
        self.phase = self.params["phase"]

        self.optimizer = self.params["optimizer"]
        self.scheduler = self.params["scheduler"]

    def get_model(self, model_name):
        model = eval(model_name)(self.phase, self.params["model_cfg"])
        model.to(self.device)
        return model
    