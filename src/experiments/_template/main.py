import sys, os
os.environ["KMP_AFFINITY"] = "disabled"
import warnings
warnings.filterwarnings("ignore")
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append("../../libs")
os.environ["HYDRA_FULL_ERROR"] = '1'
from runner import Runner

@hydra.main(config_name="config.yml")
def main(cfg : DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg)
    runner = Runner(cfg) 
    runner()
    print("success")

if __name__ == "__main__":
    main()