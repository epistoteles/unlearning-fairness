#!/bin/bash

# install prerequisites
sudo apt-get install git screen htop nvtop python3-pip libjpeg-dev zlib1g-dev -y

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
git clone https://epistoteles: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@github.com/epistoteles/unlearning-fairness.git
mv UTKFace unlearning-fairness/UTKFace
cd unlearning-fairness/ || return
git config user.email "korbinian-koch@web.de"
git config user.name "epistoteles"
# git pull https://epistoteles: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@github.com/epistoteles/unlearning-fairness.git

# install requirements
pip3 install -r requirements.txt

# install cuda
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py
rm install_gpu_driver.py

# wandb login
python3 -m wandb login