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

faces = []
for i in range(7):
    faces.append(UTKFace.denormalize(dataset.__getitem__(random.randint(0, 23700))[0]))

grid = make_grid(faces)
plt.imshow(grid.permute(1, 2, 0))
plt.show()
