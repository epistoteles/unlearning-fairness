from UTKFaceDataset import UTKFace
import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision.transforms.functional as F
from torchvision.utils import make_grid


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(15, 3))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


dataset = UTKFace()
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

white_ages = [age for age, race in zip(ages, races) if race == 0]
black_ages = [age for age, race in zip(ages, races) if race == 1]
asian_ages = [age for age, race in zip(ages, races) if race == 2]
indian_ages = [age for age, race in zip(ages, races) if race == 3]
other_ages = [age for age, race in zip(ages, races) if race == 4]
bins = np.linspace(1, 117, 117)

plt.figure(figsize=(20, 8))
plt.hist([white_ages, black_ages, asian_ages, indian_ages, other_ages],
         bins,
         stacked=True,
         label=['white', 'black', 'asian', 'indian', 'other']
         )
plt.legend(loc='upper right')
plt.savefig('plots/ages-vs-races.png')
plt.show()
