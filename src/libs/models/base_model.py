import re
from abc import ABCMeta, abstractmethod

import sys, os
sys.path.append("../")
import os.path as osp

class BaseModel(metaclass=ABCMeta):
    def __init__(self, params=None):
        self.params = params
        self.perse_params()
        self.model = self.get_model(params["model"])

    
    @abstractmethod
    def fit(self, train_X, train_y, valid_X, valid_y):
        raise NotImplementedError
    
    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abstractmethod   
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def save_weight(self, path):
        raise NotImplementedError

    @abstractmethod
    def read_weight(self, path):
        raise NotImplementedError 

    @abstractmethod
    def perse_params(self):
        raise NotImplementedError 