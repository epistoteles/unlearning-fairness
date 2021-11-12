from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer
from AgeModel import AgeModel


model = AgeModel()

logger = WandbLogger(project="age-classifier", entity='epistoteles')
lr_monitor = LearningRateMonitor(logging_interval='epoch')
trainer = Trainer(max_epochs=20, gpus=1, logger=logger, callbacks=[lr_monitor])

trainer.fit(model)
