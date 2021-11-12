import torch
from torch import nn
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import accuracy
from torch.utils.data import random_split
from UTKFaceDataset import UTKFace
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from torchvision import models

from torchvision.datasets import MNIST
from torchvision import transforms

torch.set_printoptions(linewidth=120)


class AgeModel(LightningModule):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 2
        self.classifier = nn.Linear(num_filters, num_target_classes)

        # self.conv1 = nn.Conv2d(3, 96, 7, stride=4, padding=1)
        # self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        # self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        # self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        # self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        # self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        # self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        # self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)
        # self.norm3 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        # self.fc1 = nn.Linear(18816, 512)
        # self.dropout1 = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(512, 512)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)

        # x = F.leaky_relu(self.conv1(x))
        # x = self.pool1(x)
        # x = self.norm1(x)
        # x = F.leaky_relu(self.conv2(x))
        # x = self.pool2(x)
        # x = self.norm2(x)
        # x = F.leaky_relu(self.conv3(x))
        # x = self.pool3(x)
        # x = self.norm3(x)
        # x = x.view(-1, 18816)
        # x = self.fc1(x)
        # x = F.leaky_relu(x)
        # x = self.dropout1(x)
        # x = self.fc2(x)
        # x = F.leaky_relu(x)
        # x = self.dropout2(x)
        # x = self.fc3(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000])
        return [optimizer], [scheduler]

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
        self.train_data, self.val_data = random_split(data, [len(data) - 3000, 3000])
        # data = MNIST(root='.',
        #              download=True,
        #              transform=transforms.Compose([transforms.Resize((227, 227)),
        #                                            transforms.Grayscale(3),
        #                                            transforms.ToTensor()
        #                                            ]))
        self.train_data, self.val_data = random_split(data, [len(data) - 3000, 3000])

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=256, num_workers=4)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=256, num_workers=4)
        return val_loader


model = AgeModel()

logger = WandbLogger(project="age-classifier", entity='epistoteles')
trainer = Trainer(max_epochs=100, gpus=1, logger=logger, fast_dev_run=False)

trainer.fit(model)
