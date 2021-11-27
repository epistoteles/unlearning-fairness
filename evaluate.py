from os import listdir
from os.path import join

from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy, f1

from data.UTKFaceDataset import UTKFaceDataset
from model.AgeModelResnet18 import AgeModelResnet18


run_dir = 'lazy-owl'
print(f'Evaluating run {run_dir}')

checkpoints = [f for f in listdir(join('checkpoints', run_dir))
               if f.endswith('.ckpt')]

'checkpoints/happy-bear/happy-bear-shard=2-slice=1.ckpt'
shards = [f.split('shard=')[-1].split('-')[0] for f in checkpoints]
slices = [f.split('slice=')[-1].split('.')[0] for f in checkpoints]

num_shards = max(shards) + 1
num_slices = max(slices) + 1
print(f'Found {num_shards} shards and {num_slices} slices.')

checkpoints = [f for f, s in zip(checkpoints, slices) if s == num_slices - 1]
print(f'Evaluating on following checkpoints:')
for c in checkpoints:
    print(f'   {c}')

test_data = UTKFaceDataset(split='test')
test_dataloader = DataLoader(test_data, batch_size=len(test_data), num_workers=4)
X, Y = next(iter(test_dataloader))

model = AgeModelResnet18.load_from_checkpoint(checkpoints[0])
model.eval()
logits = model(X)
for checkpoint_path in checkpoints[1:]:
    model = AgeModelResnet18.load_from_checkpoint(checkpoint_path)
    model.eval()
    logits += model(X)

loss_function = nn.CrossEntropyLoss()
loss = loss_function(logits, Y)
acc = accuracy(logits, Y)
macro_f1 = f1(logits, Y, average='macro', num_classes=7)

print(f'Loss: {loss}')
print(f'Accuracy: {acc}')
print(f'Macro F1: {macro_f1}')

