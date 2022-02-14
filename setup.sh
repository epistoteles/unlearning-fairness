#!/bin/bash

# install prerequisites
sudo apt-get install git screen htop nvtop python3-pip python3-dev libjpeg-dev zlib1g-dev gcc gfortran libopenblas-dev liblapack-dev -y

# mount bucket with UTKFace dataset
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update -y
sudo apt-get install gcsfuse -y

# unzip UTKFace to VM
mkdir data
gcsfuse utkface ./data
mkdir UTKFace
tar -xvzf 'data/UTKFace.tar.gz' -C ./
rm /data

# clone code repository
git clone https://yourusername:youraccesstoken@github.com/epistoteles/unlearning-fairness.git
mv UTKFace unlearning-fairness/UTKFace
cd unlearning-fairness/ || return
git config user.email "your.github.email@mail.com"
git config user.name "your.github.username"

# install requirements
pip3 install -r requirements.txt

# install cuda
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py
rm install_gpu_driver.py

# wandb login
python3 -m wandb login
