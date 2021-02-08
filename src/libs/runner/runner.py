from .train import Train
from .infer import Infer
from ._logging import Logging


class Runner():
    def __init__(self, param):
        self.exp_param = param["exp_param"]
        self.train_param = param["train_param"]
        self.log_param = param["log_param"]
        self.train_param.update(self.exp_param)
        self.log_param.update(self.exp_param)

    def __call__(self):
        Trainer = Train(self.train_param)
        Inferer = Infer(self.train_param)
        Logger = Logging(self.log_param)
        Trainer()
        Inferer()
        Logger()