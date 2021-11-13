from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer
from AgeModelCNN import AgeModelCNN


model = AgeModelCNN()

logger = WandbLogger(project="age-classifier", entity='epistoteles')
lr_monitor = LearningRateMonitor(logging_interval='epoch')
trainer = Trainer(max_epochs=60, gpus=1, logger=logger, callbacks=[lr_monitor])

trainer.fit(model)
