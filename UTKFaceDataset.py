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

    def __init__(self, split, image_dir='UTKFace', label='age'):
        if label not in ['age', 'gender', 'race']:
            raise ValueError(f"Unknown label type '{label}', use 'age', 'gender' or 'race'")
        if split not in ['train', 'test', 'all']:
            raise ValueError(f"Unknown split type '{label}', use 'train', 'test' or 'all'")
        self.label = label
        self.split = split
        self.image_dir = image_dir
        self.filenames = [f for f in listdir(image_dir) if
                          isfile(join(image_dir, f)) and
                          len(f.split('_')) == 4 and  # wrong names
                          f not in {'1_0_0_20170109193052283.jpg.chip.jpg',
                                    '1_0_0_20170109194120301.jpg.chip.jpg'}  # damaged 👀
                          # and random.randint(1,10) <= 2  # only keep 1/5 of data
                          # and (f.split('_')[0] != 26 or bool(random.getrandbits(1)))  # throw away 50% of 26-year-olds
                          ]
        random.seed(42)
        test_indices = random.sample(range(0, len(self.filenames)), 3000)
        if self.split == 'train':
            self.filenames = [f for i, f in enumerate(self.filenames) if i not in test_indices]
            random.shuffle(self.filenames)
        elif self.split == 'test':
            self.filenames = [f for i, f in enumerate(self.filenames) if i in test_indices]
            random.shuffle(self.filenames)
        elif self.split == 'all':
            random.shuffle(self.filenames)
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

    def __getitem__(self, index):
        """Generates one sample of data"""
        image = self.images[index]

        label = None
        if self.label == 'age':
            # age_bins = [5, 19, 24, 27, 30, 35, 40, 50, 61, 120]  # 10 equal sized
            # age_bins = [4, 12, 24, 36, 48, 60, 120]  # my own choice
            age_bins = [2, 9, 20, 27, 45, 65, 120]  # from tds blog post
            age = self.ages[index]
            for idx, age_bin in enumerate(age_bins):
                if age <= age_bin:
                    label = idx
                    break
            if label is None:  # if label was not changed above
                raise ValueError(f'Unknown age encountered: {age}')
        elif self.label == 'gender':
            label = self.genders[index]  # 0 = male, 1 = female
        elif self.label == 'race':
            label = self.races[index]  # 0 = white, 1 = black, 2 = asian, 3 = indian, 4 = others

        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.15, hue=0.05),
            transforms.RandomAffine(degrees=15, translate=None, scale=(0.85, 1.2), shear=10, fill=128),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
            transforms.RandomAutocontrast(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: x + torch.tensor(0.15, dtype=torch.float32) * torch.randn_like(x)),  # noise
        ])

        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if self.split == 'train':
            return train_transforms(image), label
        elif self.split == 'test':
            # return test_transforms(image), label
            return train_transforms(image), label

    @staticmethod
    def denormalize(image):
        """Undoes the normalization transform for viewing and plotting"""
        denorm = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        ])
        return denorm(image)
