import streamlit as st
import sys, os
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
from omegaconf import DictConfig, OmegaConf


from app_utils import parse_yaml, inference


ROOT = Path(os.getcwd()).parent # ~/src

### app contents ###
st.title('Welcome to my app!')
st.markdown("### when you uplaod image, ssd inferenced result are shown")

### sidebar settings
exps = os.listdir("../experiments/")
exps = [exp for exp in exps if exp not in ["make_ex.sh", "_template"]]
st.sidebar.markdown("# settings")

# 1.  ######
uploaded_image = st.sidebar.file_uploader("1. upload your image")
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, use_column_width=True)

# 2.  ######
selected_exp = st.sidebar.selectbox("2. select experiment you try", exps)
if selected_exp is not None:
    infer_params = parse_yaml(ROOT / "experiments" / selected_exp / "config.yml")
    st.sidebar.write(infer_params["exp_param"]["description"])
    if not os.path.exists(ROOT / "experiments" / selected_exp / "weight"):
        st.sidebar.markdown("`whoops!` :disappointed_relieved:")
        st.sidebar.markdown("`model weight doesn't exist.`")
        st.sidebar.markdown("`you may not have experiment yet.`")


# 3.  ######
image_size = st.sidebar.radio("3. decide image size (300 is better result)", [300, 512])

# 4.  ######
data_confidence_level = st.sidebar.slider('4. set confidence level (default is 0.9)',  min_value=0.0, max_value=1.0, step=0.01, value=0.9)

st.sidebar.write("Are you ready?")
do_inference = st.sidebar.button('do inference')



# Infer
if do_inference and uploaded_image is not None:
    inferenced_image = inference(
        cfg=OmegaConf.create(infer_params), 
        ROOT=ROOT, 
        image=image,
        selected_exp=selected_exp, 
        data_confidence_level=data_confidence_level,
        image_size=image_size,
    )
    inferenced_image = Image.open("img.png")
    st.image(inferenced_image, use_column_width=True)


