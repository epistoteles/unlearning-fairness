import sys
from os import listdir
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy, f1

from data.UTKFaceDataset import UTKFaceDataset
from model.AgeModelResnet18 import AgeModelResnet18


run_dir = 'lazy-owl'
print(f'Evaluating run {run_dir}')

checkpoints = [join('checkpoints', run_dir, f) for f in listdir(join('checkpoints', run_dir))
               if f.endswith('.ckpt')]

'checkpoints/happy-bear/happy-bear-shard=2-slice=1.ckpt'
shards = [int(f.split('shard=')[-1].split('-')[0]) for f in checkpoints]
slices = [int(f.split('slice=')[-1].split('.')[0]) for f in checkpoints]

num_shards = max(shards) + 1
num_slices = max(slices) + 1
print(f'Found {num_shards} shards and {num_slices} slices.')

checkpoints = [f for f, s in zip(checkpoints, slices) if s == num_slices - 1]
print(f'Evaluating on following checkpoints:')
for c in checkpoints:
    print(f'   {c}')

test_data = UTKFaceDataset(split='test')
test_dataloader = DataLoader(test_data, batch_size=128, num_workers=4)

models = []
for model_index, checkpoint_path in enumerate(checkpoints):
    models.append(AgeModelResnet18.load_from_checkpoint(checkpoint_path))

losses = []
accs = []
macro_f1s = []
lengths = []
for batch, (X, Y) in enumerate(test_dataloader):
    print(f"Evaluating batch {batch} with length {len(X)}")
    logits = torch.zeros((len(X), 7))
    for model_index, model in enumerate(models):
        print(f'   Doing inference on shard {model_index+1}/{num_shards}')
        model.eval()
        logits += torch.zeros((len(X), 7))  # model(X)
        del model
    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(logits, Y)
    acc = accuracy(logits, Y)
    macro_f1 = f1(logits, Y, average='macro', num_classes=7)
    print(f'   Overall batch metrics: loss={loss}, acc={acc}, macro_f1={macro_f1}')
    losses.append(loss)
    accs.append(acc)
    macro_f1s.append(macro_f1)
    lengths.append(len(Y))

loss = 0
acc = 0
macro_f1 = 0
for (l, a, m, length) in zip(losses, accs, macro_f1s, lengths):
    loss += l * length/len(test_data)
    accs += a * length/len(test_data)
    macro_f1 += m * length/len(test_data)

print(f'Loss: {loss}')
print(f'Accuracy: {acc}')
print(f'Macro F1: {macro_f1}')

