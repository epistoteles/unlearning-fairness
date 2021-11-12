import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
from PIL import Image
from torchvision import transforms
import random

torch.set_printoptions(linewidth=120)


class UTKFace(Dataset):
    """Characterizes the UTKFace dataset for PyTorch"""

    def __init__(self, image_dir='UTKFace'):
        self.image_dir = image_dir
        self.filenames = [f for f in listdir(image_dir) if
                          isfile(join(image_dir, f)) and
                          len(f.split('_')) == 4 and  # wrong names
                          f not in {'1_0_0_20170109193052283.jpg.chip.jpg',
                                    '1_0_0_20170109194120301.jpg.chip.jpg'} and  # damaged 👀
                          (f.split('_')[0] != 26 or bool(random.getrandbits(1)))
                          # throw away 50% of 26-year-olds (too many)
                          ]
        self.images = []
        self.ages = list(map(lambda x: int(x.split('_')[0]), self.filenames))
        self.genders = list(map(lambda x: int(x.split('_')[1]), self.filenames))
        self.races = list(map(lambda x: int(x.split('_')[2]), self.filenames))

        for f in self.filenames:
            image_path = join(self.image_dir, f)
            temp = Image.open(image_path)
            keep = temp.copy()
            self.images.append(keep)
            temp.close()

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.filenames)

    def __getitem__(self, index, label='gender'):
        """Generates one sample of data"""
        image = self.images[index]

        if label == 'age':
            age_bins = [2, 6, 14, 24, 36, 3, 40, 50, 61, 120]  # up to x years
            age = self.ages[index]
            for idx, age_bin in enumerate(age_bins):
                if age <= age_bin:
                    label = idx
                    break
            if label is 'age':  # if label was not changed above
                raise ValueError(f'Unknown age encountered: {age}')
        elif label == 'gender':
            label = self.gender[index]
        elif label == 'race':
            label = self.race[index]
        else:
            raise ValueError('label type not known')

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.0),
            # transforms.RandomAffine(degrees=15, translate=None, scale=(0.9, 1.2), shear=10, fill=0),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
            transforms.RandomAutocontrast(),
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[152.01048, 116.37661, 99.60926], std=[32.09281, 29.17887, 30.77170])
        ])

        return transform(image), label

    @staticmethod
    def denormalize(image):
        """Undoes the normalization transform for viewing and plotting"""
        denorm = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1 / 32.09281, 1 / 29.17887, 1 / 30.77170]),
            transforms.Normalize(mean=[-152.01048, -116.37661, -99.60926], std=[1., 1., 1.])
        ])
        return denorm(image)
