from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append("../libs")
from runner import Infer

def parse_yaml(file_path):
    with open(file_path, 'r') as f:
        config = OmegaConf.load(f)
    return config

def inference(cfg, ROOT, image, selected_exp, data_confidence_level, image_size):
    # cfg : DictConf
    # config setting
    cfg.exp_param.WORK_DIR = str(ROOT / "experiments" / selected_exp)
    cfg.exp_param.ROOT = str(ROOT)
    cfg.exp_param.exp_name = selected_exp
    cfg.train_param.data_confidence_level = data_confidence_level
    cfg.train_param.image_size = image_size
    cfg = OmegaConf.to_container(cfg)

    # make config for inference
    infer_params = cfg["exp_param"]
    infer_params.update(cfg["train_param"])

    Inferer = Infer(infer_params)
    inferenced_image = Inferer.infer_oneimage(image)
    return inferenced_image
