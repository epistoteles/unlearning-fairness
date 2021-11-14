from UTKFaceDataset import UTKFace
import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from torch.utils.data import ConcatDataset
from collections import Counter


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(15, 3))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


dataset = UTKFace(split='all')

plt.rcParams["savefig.bbox"] = 'tight'

# faces = []
# for i in range(7):
#     faces.append(UTKFace.denormalize(dataset.__getitem__(random.randint(0, 23700))[0]))
#
# grid = make_grid(faces)
# plt.figure(figsize=(15, 3))
# plt.imshow(grid.permute(1, 2, 0), interpolation='nearest')
# plt.savefig('plots/faces.png')
# plt.show()

ages = dataset.ages
genders = dataset.genders
races = dataset.races
#
# white_ages = [age for age, race in zip(ages, races) if race == 0]
# black_ages = [age for age, race in zip(ages, races) if race == 1]
# asian_ages = [age for age, race in zip(ages, races) if race == 2]
# indian_ages = [age for age, race in zip(ages, races) if race == 3]
# other_ages = [age for age, race in zip(ages, races) if race == 4]
# bins = np.linspace(1, 117, 117)
#
# plt.figure(figsize=(20, 8))
# plt.hist([white_ages, black_ages, asian_ages, indian_ages, other_ages],
#          bins,
#          stacked=True,
#          label=['white', 'black', 'asian', 'indian', 'other']
#          )
# plt.legend(loc='upper right')
# plt.savefig('plots/ages-vs-races.png')
# plt.show()


age_labels = []
age_bins = [2, 9, 20, 27, 45, 65, 120]
for age in ages:
    for idx, age_bin in enumerate(age_bins):
        if age <= age_bin:
            age_labels.append(idx)
            break

c = Counter(age_labels)
y = [c[key]/23000. for key in sorted(c.keys())]
fig, ax = plt.subplots()
plt.bar([*[f'{[-1, *age_bins][i]+1}-{age_bins[i]}' for i in range(len(age_bins)-1)], f'{age_bins[-2]}+'], y)
for i, v in enumerate(y):
    ax.text(i-0.3, v+0.01, f'{v:.3f}')
plt.legend(loc='upper right')
plt.savefig('plots/age-bins.png')
plt.show()
