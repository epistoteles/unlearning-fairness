import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import accuracy, f1
from data.UTKFaceDataset import UTKFaceDataset
from torchvision import models


class AgeModelResnet18(LightningModule):

    def __init__(self,
                 current_shard=0,
                 num_shards=1,
                 current_slice=0,
                 num_slices=1
                 ):
        super().__init__()

        # set SISA params
        self.current_shard = current_shard,
        self.num_shards = num_shards,
        self.current_slice = current_slice,
        self.num_slices = num_slices,

        # set hyperparams
        self.label = 'age'
        self.initial_lr = 6e-4
        if self.label == 'age':
            self.num_target_classes = 7
        elif self.label == 'race':
            self.num_target_classes = 5
        elif self.label == 'gender':
            self.num_target_classes = 2
        self.loss_weights = torch.Tensor(
                [1 / 0.07 / 7, 1 / 0.063 / 7, 1 / 0.079 / 7, 1 / 0.242 / 7,
                 1 / 0.332 / 7, 1 / 0.17 / 7, 1 / 0.074 / 7]).cuda()  # for weighting loss of unbalanced classes

        # use first 3 parts as frozen feature extractor to save training time
        pretrained_layers = list(models.resnet18(pretrained=True).children())[:-7]
        self.feature_extractor1 = nn.Sequential(*pretrained_layers)

        # use middle part of a pretrained resnet
        pretrained_layers = list(models.resnet18(pretrained=True).children())[3:-3]
        self.feature_extractor2 = nn.Sequential(*pretrained_layers)

        # use later part of a randomly initialized resnet
        random_layers = list(models.resnet18(pretrained=False).children())[7:-1]
        self.feature_extractor3 = nn.Sequential(*random_layers)

        # add custom fully connected layers at the end
        self.relu = nn.ReLU()
        self.fc1 = nn.LazyLinear(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.1)
        self.classifier = nn.Linear(128, self.num_target_classes)

        # filled in setup()
        self.train_data = None
        self.val_data = None

    def forward(self, x):
        self.feature_extractor1.eval()
        with torch.no_grad():
            x = self.feature_extractor1(x)
        x = self.feature_extractor2(x)
        x = self.feature_extractor3(x).flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 9, 12], gamma=0.2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss_epoch"}

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)  # weight=self.loss_weights -> worse
        loss = loss_function(logits, y)
        acc = accuracy(logits, y)
        macro_f1 = f1(logits, y, average='macro', num_classes=7)
        conf_matrix = ConfusionMatrix(num_classes=self.num_target_classes, normalize='true')
        self.log("batch_acc", acc, prog_bar=True)
        self.log("batch_macro_f1", macro_f1)
        return {'loss': loss, 'accuracy': acc, 'macro_f1': macro_f1, 'conf_matrix': conf_matrix}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(logits, y)
        acc = accuracy(logits, y)
        macro_f1 = f1(logits, y, average='macro', num_classes=7)
        conf_matrix = ConfusionMatrix(num_classes=self.num_target_classes, normalize='true')
        self.log("batch_acc", acc, prog_bar=True)
        self.log("batch_macro_f1", macro_f1)
        return {'loss': loss, 'accuracy': acc, 'macro_f1': macro_f1, 'conf_matrix': conf_matrix}

    def training_epoch_end(self, train_step_outputs):
        avg_train_loss = torch.tensor([x['loss'] for x in train_step_outputs]).mean()
        avg_train_acc = torch.tensor([x['accuracy'] for x in train_step_outputs]).mean()
        avg_train_macro_f1 = torch.tensor([x['macro_f1'] for x in train_step_outputs]).mean()
        self.log("train/loss_epoch", avg_train_loss)
        self.log("train/acc_epoch", avg_train_acc)
        self.log("train/macro_f1_epoch", avg_train_macro_f1)

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['accuracy'] for x in val_step_outputs]).mean()
        avg_val_macro_f1 = torch.tensor([x['macro_f1'] for x in val_step_outputs]).mean()
        self.log("val/loss_epoch", avg_val_loss)
        self.log("val/acc_epoch", avg_val_acc)
        self.log("val/macro_f1_epoch", avg_val_macro_f1)
        return {'val_loss': avg_val_loss, 'val_acc': avg_val_acc}

    def setup(self, stage):
        self.train_data = UTKFaceDataset(split='train', label=self.label,
                                         current_shard=self.current_shard,
                                         num_shards=self.num_shards,
                                         current_slice=self.current_slice,
                                         num_slices=self.num_slices)
        self.val_data = UTKFaceDataset(split='test', label=self.label,
                                       current_shard=self.current_shard,
                                       num_shards=self.num_shards,
                                       current_slice=self.current_slice,
                                       num_slices=self.num_slices)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_data, batch_size=128, num_workers=4)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data, batch_size=512, num_workers=4)
        return val_loader
