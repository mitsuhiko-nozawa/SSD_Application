from abc import ABCMeta, abstractmethod
import os
import os.path as osp
class BaseManager(metaclass=ABCMeta):
    def __init__(self, params):
        self.params = params
        self.ROOT = params["ROOT"]
        self.WORK_DIR = params["WORK_DIR"]
        self.raw_dirname = params["raw_dirname"] 
        self.data_path = osp.join(self.ROOT, "input", self.raw_dirname)
        self.val_preds_path = osp.join(self.WORK_DIR, "val_preds")
        self.preds_path = osp.join(self.WORK_DIR, "preds")
        self.weight_path = osp.join(self.WORK_DIR, "weight")
        self.seeds = params["seeds"]
        self.debug = params["debug"]
        self.voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

        if not osp.exists(self.val_preds_path): os.mkdir(self.val_preds_path)
        if not osp.exists(self.weight_path): os.mkdir(self.weight_path)
        if not osp.exists(self.preds_path): os.mkdir(self.preds_path)

    def get(self, key):
        try:
            return self.params[key]
        except:
            raise ValueError(f"No such value in params, {key}")
        
    @abstractmethod
    def __call__(self):
        raise NotImplementedError