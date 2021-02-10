import streamlit as st
import sys, os
sys.path.append("../libs/")
from pathlib import Path
from PIL import Image

import hydra
from omegaconf import DictConfig, OmegaConf

from runner.infer import Infer

### app contents ###
ROOT = Path(os.getcwd()).parent

st.title('Welcome to my app!')
st.write("when you uplaod image, ssd inferenced results are shown")

### sidebar settings
exps = os.listdir("../experiments/")
exps = [exp for exp in exps if exp not in ["make_ex.sh", "_template"]]

st.sidebar.markdown("# settings")
selected_exp = st.sidebar.selectbox('select experiment you try', exps)
do_inference = st.sidebar.button('do inference')

### image uploading
uploaded_image = st.file_uploader("")
if uploaded_image is not None:
    
    image = Image.open(uploaded_image)
    st.image(image)

if do_inference and uploaded_image is not None:
    @hydra.main(config_path=f"../experiments/{selected_exp}", config_name="config.yml", strict=False)
    def inference(cfg : DictConfig) -> None:
        # config setting
        cfg.exp_param.WORK_DIR = str(ROOT / "experiments" / selected_exp)
        cfg.exp_param.ROOT = str(ROOT)
        cfg.exp_param.exp_name = selected_exp
        cfg.train_param.data_confidence_level = 0.9
        cfg = OmegaConf.to_container(cfg)

        # make config for inference
        infer_params = cfg["exp_param"]
        infer_params.update(cfg["train_param"])
        #st.write(infer_params)

        Inferer = Infer(infer_params)
        inferenced_image = Inferer.infer_oneimage(image)
        inferenced_image = Image.open("img.png")
        st.image(inferenced_image, use_column_width=True)

    inference()