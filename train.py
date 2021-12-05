from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer
from model.AgeModelResnet18 import AgeModelResnet18
import itertools
import wandb
import argparse
from utils import random_run_name


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('num_shards', type=int, nargs=1, help='number of shards')
parser.add_argument('num_slices', type=int, nargs=1, help='number of slices')
parser.add_argument('strategy', type=str, nargs=1, help='strategy: random-balanced or sorted-balanced')
parser.add_argument('put_in', type=str, nargs=1, help='strategy to distribute samples: slice or shard')
args = parser.parse_args()

num_shards = args.num_shards[0]
num_slices = args.num_slices[0]
strategy = args.strategy[0]
put_in = args.put_in[0]

# run_name = random_run_name()
run_name = f'{strategy}-{put_in}'

for i in range(5):
    print(f"Starting experiment run {run_name}-{i}/10 ...")
    print(f"Shards: {num_shards}, Slices: {num_slices}")
    print(f"Strategy: {strategy} with put_in {put_in}")

    for current_shard, current_slice in itertools.product(range(num_shards), range(num_slices)):
        if current_slice == 0:  # first slice
            model = AgeModelResnet18(current_shard=current_shard,
                                     num_shards=num_shards,
                                     current_slice=current_slice,
                                     num_slices=num_slices)
        else:  # later slices with pretrained checkpoints
            model = AgeModelResnet18.load_from_checkpoint(
                f'checkpoints/{run_name}/{run_name}-shard={current_shard}-slice={current_slice - 1}.ckpt',
                current_shard=current_shard,
                num_shards=num_shards,
                current_slice=current_slice,
                num_slices=num_slices)

        logger = WandbLogger(project="age-classifier",
                             entity='epistoteles',
                             id=f'{run_name}-{i}/10-shard-{current_shard}-slice-{current_slice}',
                             group=f'{run_name}')
        lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')
        checkpoint_cb = ModelCheckpoint(save_top_k=1,
                                        monitor=None,  # saves checkpoint for last epoch
                                        # monitor='val/macro_f1_epoch',  # saves checkpoint for best epoch
                                        # mode='max',  # necessary when tracking macro F1
                                        dirpath=f"checkpoints/{run_name}/",
                                        filename=f"{run_name}-shard={current_shard}-slice={current_slice}",
                                        save_weights_only=True)
        trainer = Trainer(max_epochs=15, gpus=1, logger=logger, callbacks=[lr_monitor_cb, checkpoint_cb])

        trainer.fit(model)
        wandb.finish()
