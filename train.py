from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer
from AgeModelCNN import AgeModelCNN
from AgeModelVGG import AgeModelVGG
from AgeModelResnet50 import AgeModelResnet50
from AgeModelResnet18 import AgeModelResnet18


model = AgeModelResnet18()

logger = WandbLogger(project="age-classifier", entity='epistoteles')
lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')
checkpoint_cb = ModelCheckpoint(monitor="val/macro_f1_epoch",
                                dirpath="checkpoints/run1/",
                                filename="agemodel-shard=1-{epoch:02d}",
                                save_top_k=1,
                                mode='max')
trainer = Trainer(max_epochs=15, gpus=1, logger=logger, callbacks=[lr_monitor_cb, checkpoint_cb])

trainer.fit(model)
