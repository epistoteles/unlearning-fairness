
! nvidia-smi

import torch
torch.set_printoptions(linewidth=120)

from google.colab import drive, files
drive.mount('/content/gdrive')
!cp 'gdrive/My Drive/Colab Notebooks/Unlearning Fairness/UTKFaceDataset.py' '/content/UTKFaceDataset.py'
![ -d UTKFace ] || tar -xvzf 'gdrive/My Drive/Colab Notebooks/Unlearning Fairness/UTKFace.tar.gz'

from importlib import reload
import UTKFaceDataset
reload(UTKFaceDataset)
from UTKFaceDataset import UTKFace
import matplotlib.pyplot as plt

%%capture
! pip install --upgrade pip
! pip install pytorch-lightning
! pip install torchmetrics
! pip install shortuuid==1.0.1
! pip install wandb

from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import accuracy
from torch.utils.data import random_split


class AgeModel(LightningModule):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 96, 7, stride = 4, padding = 1)
    self.pool1 = nn.MaxPool2d(3, stride = 2, padding = 1)
    self.norm1 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)
    self.conv2 = nn.Conv2d(96, 256, 5, stride = 1, padding = 2)
    self.pool2 = nn.MaxPool2d(3, stride = 2, padding = 1)
    self.norm2 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)
    self.conv3 = nn.Conv2d(256, 384, 3, stride = 1, padding = 1)
    self.pool3 = nn.MaxPool2d(3, stride = 2, padding = 1)
    self.norm3 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)
    self.fc1 = nn.Linear(18816, 512)
    self.dropout1 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(512, 512)
    self.dropout2 = nn.Dropout(0.5)
    self.fc3 = nn.Linear(512, 10)

  def forward(self, x):
    x = F.leaky_relu(self.conv1(x))
    x = self.pool1(x)
    x = self.norm1(x)
    x = F.leaky_relu(self.conv2(x))
    x = self.pool2(x)
    x = self.norm2(x)
    x = F.leaky_relu(self.conv3(x))
    x = self.pool3(x)
    x = self.norm3(x)
    x = x.view(-1, 18816)
    x = self.fc1(x)
    x = F.leaky_relu(x)
    x = self.dropout1(x)
    x = self.fc2(x)
    x = F.leaky_relu(x)
    logits = self.dropout2(x)
    return logits

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000])
    return ([optimizer], [scheduler])

  def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(logits, y)
    acc = accuracy(logits, y)
    conf_matrix = ConfusionMatrix(num_classes=10, normalize='true')
    self.log("train_acc", acc, prog_bar=True)
    return {'loss': loss, 'accuracy': acc, 'conf_matrix': conf_matrix}

  def validation_step(self, batch, batch_idx):
    results = self.training_step(batch, batch_idx)
    return results

  def training_epoch_end(self, train_step_outputs):
    avg_train_loss = torch.tensor([x['loss'] for x in train_step_outputs]).mean()
    avg_train_acc = torch.tensor([x['accuracy'] for x in train_step_outputs]).mean()
    self.log("train/loss_epoch", avg_train_loss, prog_bar=True)    
    self.log("train/acc_epoch", avg_train_acc, prog_bar=True)    

  def validation_epoch_end(self, val_step_outputs):
    avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
    avg_val_acc = torch.tensor([x['accuracy'] for x in val_step_outputs]).mean()
    self.log("val/loss_epoch", avg_val_loss, prog_bar=True)    
    self.log("val/acc_epoch", avg_val_acc, prog_bar=True)    
    return {'val_loss': avg_val_loss, 'val_acc': avg_val_acc}

  def setup(self, stage):
    data = UTKFace()
    self.train_data, self.val_data = random_split(data, [len(data)-3000, 3000])

  def train_dataloader(self):
    train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=50, num_workers=2)
    return train_loader

  def val_dataloader(self):
    val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=50, num_workers=2)
    return val_loader


model = AgeModel()

! wandb login

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

logger = WandbLogger(project="age-classifier", entity='epistoteles')
trainer = Trainer(max_epochs=200, gpus=1, logger=logger, fast_dev_run=False)

trainer.fit(model)

# import numpy as np
# import matplotlib.pyplot as plt
# import random
# import torchvision.transforms.functional as F
# from torchvision.utils import make_grid


# dataset = UTKFace()  # only used for plotting, re-defined in model

# plt.rcParams["savefig.bbox"] = 'tight'

# def show(imgs):
#     if not isinstance(imgs, list):
#         imgs = [imgs]
#     fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(15,3))
#     for i, img in enumerate(imgs):
#         img = img.detach()
#         img = F.to_pil_image(img)
#         axs[0, i].imshow(np.asarray(img))
#         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
# faces = []
# for i in range(7):
#   faces.append(UTKFace.denormalize(dataset.__getitem__(random.randint(0,23700))[0]))

# grid = make_grid(faces)
# show(grid)

# data = UTKFace()
# x, y = data.__getitem__(0)
# print(y)
