import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import accuracy
from data.UTKFaceDataset import UTKFaceDataset
from torchvision import models


class AgeModelVGG(LightningModule):
    def __init__(self):
        super().__init__()

        # set hyperparams
        self.label = 'age'
        self.initial_lr = 1e-4
        self.milestones = list(range(2, 11, 2))  # try [2, 4, 6, 10, 15, 18] ?
        self.gamma = 0.6
        if self.label == 'age':
            self.num_target_classes = 7
        elif self.label == 'race':
            self.num_target_classes = 5
        elif self.label == 'gender':
            self.num_target_classes = 2

        # init a pretrained resnet
        backbone = models.vgg16(pretrained=False)
        self.feature_extractor = list(backbone.children())[0]
        self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=self.num_target_classes)
        )

        # filled in setup()
        self.train_data = None
        self.val_data = None

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = self.classifier(x)
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
        self.train_data = UTKFaceDataset(split='train', label=self.label)
        self.val_data = UTKFaceDataset(split='test', label=self.label)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_data, batch_size=64, num_workers=4)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data, batch_size=512, num_workers=4)
        return val_loader
