#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

# python install
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN update-alternatives --set python /usr/bin/python3.7
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

# nano wget curl
RUN apt-get -y install vim nano wget curl

# opencv lib packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# pip install
RUN pip install -U pip
RUN pip install --upgrade pip
RUN pip install pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip install torch
#RUN pip install torchvision
RUN pip install opencv-python==4.4.0.46
RUN pip install jupyter jupyter-contrib-nbextensions jupyter_nbextensions_configurator
RUN pip install pytest
RUN pip install nose
RUN pip install Flask
RUN pip install tqdm
RUN pip install torchsummary
RUN pip install hydra-core
RUN pip install mlflow

# duser setting
ARG USER_ID
ARG GROUP_ID
#RUN addgroup --gid $GROUP_ID duser && \
#    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID duser && \
#    adduser duser sudo && \
#    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
