import pickle
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
import pickle

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('run_dir', type=str, nargs=1, help='the run you want to evaluate')
args = parser.parse_args()
run_dir = args.run_dir[0]
print(f'Evaluating run {run_dir}')

checkpoints = [join('checkpoints', run_dir, f) for f in listdir(join('checkpoints', run_dir))
               if f.endswith('.ckpt')]

# example: 'checkpoints/happy-bear/happy-bear-sleeps-shard=2-slice=1.ckpt'
shards = [int(f.split('shard=')[-1].split('-')[0]) for f in checkpoints]
slices = [int(f.split('slice=')[-1].split('.')[0]) for f in checkpoints]

num_shards = max(shards) + 1
num_slices = max(slices) + 1
print(f'Found {num_shards} shards and {num_slices} slices.')

checkpoints = [f for f, s in zip(checkpoints, slices) if s == num_slices - 1]
checkpoints = sorted(checkpoints, key=lambda x: (int(x.split('-shard=')[0].split('of')[-1]),  # Xof5
                                                 int(x.split('shard=')[-1].split('-')[0])))   # shard X
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
race_names = ['white', 'black', 'asian', 'indian', 'other', 'random']
test_groups = list(itertools.product(race_names, age_bin_names))

loss_function = nn.CrossEntropyLoss()
losses = []
top1_accs = []
top2_accs = []
macro_f1s = []
lengths = []
ys = torch.Tensor().cuda()
y_preds = torch.Tensor().cuda()

result_dict = {}

for batch, (X, Y) in enumerate(test_dataloader):
    X = X.to(device)
    Y = Y.to(device)
    print(f"Evaluating subgroup {test_groups[batch]} with length {len(X)}")
    logits = torch.zeros((len(X), 7)).to(device)
    for model_index, model in enumerate(models):
        with torch.no_grad():
            model.to(device)
            model.eval()
            temp_logits = model(X)
            # logits += torch.rand((len(X), 7))
        loss = loss_function(temp_logits, Y)
        top1_acc = accuracy(temp_logits, Y)
        top2_acc = accuracy(temp_logits, Y, top_k=2)
        macro_f1 = f1(temp_logits, Y, average='macro', num_classes=7)
        print(f'   Shard {model_index+1}/{num_shards} metrics: loss={loss:.4f}, top1_acc={top1_acc:.4f}, top2_acc={top2_acc:.4f}, macro_f1={macro_f1:.4f}')
        logits += temp_logits
    loss = loss_function(logits, Y)
    top1_acc = accuracy(logits, Y)
    top2_acc = accuracy(logits, Y, top_k=2)
    macro_f1 = f1(logits, Y, average='macro', num_classes=7)
    print(f'   Overall subgroup metrics: loss={loss:.4f}, top1_acc={top1_acc:.4f}, top2_acc={top2_acc:.4f}, macro_f1={macro_f1:.4f}')
    print(f"          True = {Y}")
    print(f"     Predicted = {torch.argmax(logits, dim=1)}")
    losses.append(loss)
    top1_accs.append(top1_acc)
    top2_accs.append(top2_acc)
    macro_f1s.append(macro_f1)
    lengths.append(len(Y))
    ys = torch.cat((ys, Y), dim=0)
    y_preds = torch.cat((y_preds, torch.argmax(logits, dim=1)), dim=0)
    if batch % 7 == 6:
        print('-'*35)
        print(f"Average metrics for race '{test_groups[batch][0]}':")
        print(f'   Loss: {sum(losses[-7:])/7:.4f}')
        print(f'   Top-1 Accuracy: {sum(top1_accs[-7:]) / 7:.4f}')
        print(f'   Top-2 Accuracy: {sum(top2_accs[-7:]) / 7:.4f}')
        print(f"   Macro F1: {f1(y_preds[-9*7:].int(), ys[-9*7:].int(), average='macro', num_classes=7):.4f}")
        print('-'*35)
        result_dict[test_groups[batch][0]] = (sum(top1_accs[-7:]) / 7, sum(top2_accs[-7:]) / 7)  # {race: (top1_acc, top2_acc)}

loss = 0
top1_acc = 0
top2_acc = 0
macro_f1 = 0
print(top1_accs)
print(top2_accs)
for (l, t1a, t2a, m, length) in zip(losses[:-7], top1_accs[:-7], top2_accs[:-7], macro_f1s[:-7], lengths[:-7]):
    loss += l.cpu().numpy() * length / (len(test_data)-7*9)
    top1_acc += t1a.cpu().numpy() * length / (len(test_data)-7*9)
    top2_acc += t2a.cpu().numpy() * length / (len(test_data)-7*9)
    macro_f1 += m.cpu().numpy() * length / (len(test_data)-7*9)

print('-' * 35)
print(f'Results on all 5 races:')
print(f'   Loss: {loss:.4f}')
print(f'   Top-1 Accuracy: {top1_acc:.4f}')
print(f'   Top-2 Accuracy: {top2_acc:.4f}')
print(f"   Macro F1: {f1(y_preds.int(), ys.int(), average='macro', num_classes=7):.4f}")
print('-' * 35)

result_dict['all_races'] = (top1_acc, top2_acc)

pickle.dump(result_dict, open(f"summaries/{run_dir}.pickle", "wb"))
