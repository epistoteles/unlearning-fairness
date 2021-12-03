import sys
from os import listdir
from os.path import join
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy, f1
from data.UTKFaceDataset import UTKFaceDataset
from model.AgeModelResnet18 import AgeModelResnet18
import argparse
import itertools

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('run_dir', type=str, nargs=1, help='the run you want to evaluate')
args = parser.parse_args()
run_dir = args.run_dir[0]
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

test_data = UTKFaceDataset(split='test', label='age', num_shards=5, num_slices=2, current_shard=0, current_slice=0)
test_dataloader = DataLoader(test_data, batch_size=9, num_workers=4,
                             sampler=torch.utils.data.SequentialSampler(test_data))

device = torch.device('cuda')
models = []
for model_index, checkpoint_path in enumerate(checkpoints):
    models.append(AgeModelResnet18.load_from_checkpoint(checkpoint_path))

age_bins = [2, 9, 20, 27, 45, 65, 120]
age_bin_names = [*[f'{[-1, *age_bins][i]+1}-{age_bins[i]}' for i in range(len(age_bins)-1)], f'{age_bins[-2]}+']
race_names = ['white', 'black', 'asian', 'indian', 'other']
test_groups = list(itertools.product(race_names, age_bin_names))

loss_function = nn.CrossEntropyLoss()
losses = []
accs = []
macro_f1s = []
lengths = []

for batch, (X, Y) in enumerate(test_dataloader):
    X = X.to(device)
    Y = Y.to(device)
    print(f"Evaluating subgroup {test_groups[batch]} with length {len(X)}")
    logits = torch.zeros((len(X), 7)).to(device)
    for model_index, model in enumerate(models):
        print(f'   Doing inference on shard {model_index+1}/{num_shards}')
        with torch.no_grad():
            model.to(device)
            model.eval()
            temp_logits = model(X)
            # logits += torch.rand((len(X), 7))
        loss = loss_function(temp_logits, Y)
        acc = accuracy(temp_logits, Y)
        macro_f1 = f1(temp_logits, Y, average='macro', num_classes=7)
        print(f'      Shard metrics: loss={loss}, acc={acc}, macro_f1={macro_f1}')
        logits += temp_logits
    loss = loss_function(logits, Y)
    acc = accuracy(logits, Y)
    macro_f1 = f1(logits, Y, average='macro', num_classes=7)
    print(f'   Overall subgroup metrics: loss={loss}, acc={acc}, macro_f1={macro_f1}')
    losses.append(loss)
    accs.append(acc)
    macro_f1s.append(macro_f1)
    lengths.append(len(Y))
    if batch % 7 == 6:
        print('-'*30)
        print(f"Average metrics for race '{test_groups[batch][0]}':")
        print(f'   Loss: {sum(losses[-7:])/7}')
        print(f'   Accuracy: {sum(accs[-7:])/7}')
        print(f'   Macro F1: {sum(macro_f1s[-7:])/7}')
        print('-'*30)

loss = 0
acc = 0
macro_f1 = 0
for (l, a, m, length) in zip(losses, accs, macro_f1s, lengths):
    loss += l.cpu().numpy() * length/len(test_data)
    acc += a.cpu().numpy() * length/len(test_data)
    macro_f1 += m.cpu().numpy() * length/len(test_data)

print('-' * 30)
print(f'Overall results:')
print(f'   Loss: {loss}')
print(f'   Accuracy: {acc}')
print(f'   Macro F1: {macro_f1}')
print('-' * 30)
