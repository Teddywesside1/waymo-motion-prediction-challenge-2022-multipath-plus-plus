#!/bin/bash
pip install --upgrade pip
pip install waymo-open-dataset-tf-2-11-0==1.6.0
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torchscatter