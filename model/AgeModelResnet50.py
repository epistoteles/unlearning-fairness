import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import accuracy
from data.UTKFaceDataset import UTKFaceDataset
from torchvision import models


class AgeModelResnet50(LightningModule):
    def __init__(self):
        super().__init__()

        # set hyperparams
        self.label = 'age'
        self.initial_lr = 5e-4
        self.milestones = list(range(2, 9, 2))
        self.gamma = 0.5
        if self.label == 'age':
            self.num_target_classes = 7
        elif self.label == 'race':
            self.num_target_classes = 5
        elif self.label == 'gender':
            self.num_target_classes = 2

        # init a pretrained resnet
        backbone = models.resnet50(pretrained=True)
        frozen_layers = list(backbone.children())[:-5]
        self.frozen_feature_extractor = nn.Sequential(*frozen_layers)

        # use the last few layers with trainable parameters
        trainable_layers = list(backbone.children())[5:-1]
        self.trainable_feature_extractor = nn.Sequential(*trainable_layers)

        # add custom fully connected layers at the end
        num_filters = backbone.fc.in_features
        self.dropout1 = nn.Dropout(0.5)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(num_filters, num_filters//2)
        self.dropout2 = nn.Dropout(0.8)
        self.fc2 = nn.Linear(num_filters//2, num_filters//4)
        self.dropout3 = nn.Dropout(0.7)
        self.classifier = nn.Linear(num_filters//4, self.num_target_classes)

        # filled in setup()
        self.train_data = None
        self.val_data = None

    def forward(self, x):
        self.frozen_feature_extractor.eval()
        with torch.no_grad():
            frozen_representations = self.frozen_feature_extractor(x)
        learned_representations = self.trainable_feature_extractor(frozen_representations).flatten(1)
        x = self.dropout1(learned_representations)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss_epoch"}

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
        train_loader = DataLoader(self.train_data, batch_size=128, num_workers=4)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data, batch_size=512, num_workers=4)
        return val_loader
