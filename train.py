import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import accuracy
from torch.utils.data import random_split
from UTKFaceDataset import UTKFace
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from torchvision import models

torch.set_printoptions(linewidth=120)


class AgeModel(LightningModule):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(pretrained=True)
        frozen_layers = list(backbone.children())[:-6]
        self.frozen_feature_extractor = nn.Sequential(*frozen_layers)

        # use the last few layers with trainable parameters
        trainable_layers = list(backbone.children())[4:-1]
        self.trainable_feature_extractor = nn.Sequential(*trainable_layers)

        # add a custom fully connected layer at the end
        num_filters = backbone.fc.in_features
        num_target_classes = 2
        self.classifier = nn.Linear(num_filters, num_target_classes)

        # filled in setup()
        self.train_data = None
        self.val_data = None

    def forward(self, x):
        self.frozen_feature_extractor.eval()
        with torch.no_grad():
            frozen_representations = self.frozen_feature_extractor(x)
        learned_representations = self.trainable_feature_extractor(frozen_representations).flatten(1)
        x = self.classifier(learned_representations)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20])
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

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=64, num_workers=4)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=512, num_workers=4)
        return val_loader


model = AgeModel()

logger = WandbLogger(project="age-classifier", entity='epistoteles')
trainer = Trainer(max_epochs=30, gpus=1, logger=logger, fast_dev_run=False)

trainer.fit(model)
