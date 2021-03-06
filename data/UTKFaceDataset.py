import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
from PIL import Image
from torchvision import transforms
import random
import utils
from data.Face import Face
from sampler.Sampler import Sampler

class UTKFaceDataset(Dataset):
    """Characterizes the UTKFace dataset for PyTorch"""

    def __init__(self,
                 split,
                 num_shards=1,
                 current_shard=0,
                 num_slices=1,
                 current_slice=0,
                 image_dir='UTKFace',
                 label='age'
                 ):
        if type(num_shards) is tuple:
            num_shards = num_shards[0]
        if type(num_slices) is tuple:
            num_slices = num_slices[0]
        if type(current_shard) is tuple:
            current_shard = current_shard[0]
        if type(current_slice) is tuple:
            current_slice = current_slice[0]
        if label not in ['age', 'gender', 'race']:
            raise ValueError(f"Unknown label type '{label}', use 'age', 'gender' or 'race'")
        if split not in ['train', 'test', 'all']:
            raise ValueError(f"Unknown split type '{label}', use 'train', 'test' or 'all'")
        if current_shard < 0 or current_shard >= num_shards:
            raise ValueError(f"Invalid shard number: {current_shard}")
        if current_slice < 0 or current_slice >= num_slices:
            raise ValueError(f"Invalid slice number: {current_slice}")

        self.label = label
        self.split = split
        self.image_dir = image_dir
        self.current_shard = current_shard,
        self.num_shards = num_shards,
        self.current_slice = current_slice,
        self.num_slices = num_slices,

        filenames = [f for f in listdir(image_dir) if
                     isfile(join(image_dir, f)) and
                     len(f.split('_')) == 4 and  # wrong names
                     f not in {'1_0_0_20170109193052283.jpg.chip.jpg',
                               '1_0_0_20170109194120301.jpg.chip.jpg'}]  # damaged ????

        sampler = Sampler(strategy=['gender', 'race'])
        self.faces = []
        for f in filenames:
            image_path = join(self.image_dir, f)
            temp = Image.open(image_path)
            keep = temp.copy()
            age = int(f.split('_')[0])
            gender = int(f.split('_')[1])
            race = int(f.split('_')[2])
            face = Face(image=keep, age=age, gender=gender, race=race, filename=f, sampler=sampler)
            self.faces.append(face)
            temp.close()

        indices = utils.get_indices(self.faces,
                                    num_shards=self.num_shards,
                                    num_slices=self.num_slices,
                                    strategy='random-balanced',
                                    put_in='shard',
                                    random_seed=random.randint(0, 1000))

        if self.split == 'train':
            self.faces = [self.faces[i] for i in indices[current_shard * num_slices + current_slice]]
            random.shuffle(self.faces)
        elif self.split == 'test':
            self.faces = [self.faces[i] for i in indices[-1]]
        elif self.split == 'all':
            random.shuffle(self.faces)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.faces)

    def __getitem__(self, index):
        """Generates one sample of data"""
        image = self.faces[index].image

        label = None
        if self.label == 'age':
            label = self.faces[index].age_bin
        elif self.label == 'gender':
            label = self.faces[index].gender  # 0 = male, 1 = female
        elif self.label == 'race':
            label = self.faces[index].race  # 0 = white, 1 = black, 2 = asian, 3 = indian, 4 = others

        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.15, hue=0.05),
            transforms.RandomAffine(degrees=15, translate=None, scale=(0.85, 1.2), shear=10, fill=128),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
            transforms.RandomAutocontrast(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: x + torch.tensor(0.15, dtype=torch.float32) * torch.randn_like(x)),  # noise
        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if self.split == 'train':
            return train_transforms(image), label
        elif self.split == 'test':
            return test_transforms(image), label

    @staticmethod
    def denormalize(image):
        """Undoes the normalization transform for viewing and plotting"""
        denorm = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        ])
        return denorm(image)
