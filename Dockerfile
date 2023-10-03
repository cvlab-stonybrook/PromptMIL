FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN apt-get update && apt install -y htop zsh openslide-tools vim git unzip zip libturbojpeg libvips dos2unix ffmpeg libsm6 libxext6 && apt-get clean

RUN conda install h5py numba ninja -y

RUN pip install pandas openslide-python opencv-contrib-python kornia gpustat pytorch-lightning lightning torchmetrics hydra-core albumentations timm torchstain submitit wandb tqdm tensorboardX matplotlib scipy scikit-image scikit-learn jpeg4py pyvips pyyaml yacs einops psutil simplejson termcolor terminaltables codecov flake8 isort pytest pytest-cov pytest-runner xdoctest kwarray contextlib2 absl-py pycocotools onnxruntime onnx python-box segmentation_models_pytorch

