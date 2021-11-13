import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import accuracy
from UTKFaceDataset import UTKFace


class AgeModelCNN(LightningModule):
    def __init__(self):
        super().__init__()

        # set hyperparams
        self.label = 'age'
        self.initial_lr = 1e-3
        self.milestones = []  # try [2, 4, 6, 10, 15, 18] ?
        self.gamma = 0.6
        if self.label == 'age':
            self.num_target_classes = 7
        elif self.label == 'race':
            self.num_target_classes = 5
        elif self.label == 'gender':
            self.num_target_classes = 2

        # build custom CNN
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool4 = nn.AvgPool2d(kernel_size=20)
        self.flatten5 = nn.Flatten()
        self.fc5 = nn.Linear(in_features=256, out_features=132)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(in_features=132, out_features=self.num_target_classes)
        self.softmax = nn.Softmax()

        # filled in setup()
        self.train_data = None
        self.val_data = None

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = self.relu5(self.fc5(self.flatten5(x)))
        x = self.softmax(self.fc6(x))
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(logits, y)
        acc = accuracy(logits, y)
        conf_matrix = ConfusionMatrix(num_classes=self.num_target_classes, normalize='true')
        self.log("batch_acc", acc, prog_bar=True)
        return {'loss': loss, 'accuracy': acc, 'conf_matrix': conf_matrix}

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        return results

    def training_epoch_end(self, train_step_outputs):
        avg_train_loss = torch.tensor([x['loss'] for x in train_step_outputs]).mean()
        avg_train_acc = torch.tensor([x['accuracy'] for x in train_step_outputs]).mean()
        self.log("train/loss_epoch", avg_train_loss)
        self.log("train/acc_epoch", avg_train_acc)

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['accuracy'] for x in val_step_outputs]).mean()
        self.log("val/loss_epoch", avg_val_loss)
        self.log("val/acc_epoch", avg_val_acc)
        return {'val_loss': avg_val_loss, 'val_acc': avg_val_acc}

    def setup(self, stage):
        data = UTKFace(label=self.label)
        self.train_data, self.val_data = random_split(data, [len(data) - 3000, 3000])

    def train_dataloader(self):
        train_loader = DataLoader(self.train_data, batch_size=128, num_workers=4)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data, batch_size=512, num_workers=4)
        return val_loader
