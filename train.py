from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from AgeModel import AgeModel

# torch.set_printoptions(linewidth=120)

model = AgeModel()

logger = WandbLogger(project="age-classifier", entity='epistoteles')
trainer = Trainer(max_epochs=30, gpus=1, logger=logger, fast_dev_run=False)

trainer.fit(model)
