import random

import torch


def random_split(dataset, lengths, random_seed):
    return list(map(list, torch.utils.data.random_split(dataset,
                                                        lengths,
                                                        generator=torch.Generator().manual_seed(random_seed))))


def split(a, n):
    k, m = divmod(len(a), n)  # len(a)=43, n=4, k=10, m=3
    return [k + int(i < m) for i in range(n)]


def get_counts(length, num_shards, num_slices, test=0.1, return_indices=False):
    if type(num_shards) is tuple:
        num_shards = num_shards[0]
    if type(num_slices) is tuple:
        num_slices = num_slices[0]
    test_len = int(length * test)
    train_val_len = length - test_len
    n = num_shards * num_slices
    d, m = divmod(train_val_len, n)
    lengths = [*[d + int(i < m) for i in range(n)],
               test_len]  # [shard_1_slice_1, shard_1_slice_2, shard_2_slice_1, ..., test]
    if not return_indices:
        return lengths
    else:
        return random_split(range(length), lengths, random_seed=42)


def random_run_name():
    animals = ['bear', 'tiger', 'panther', 'scorpion', 'owl', 'salmon']
    adjectives = ['lazy', 'happy', 'green', 'blue', 'hungry', 'warm', 'royal', 'bored', 'marble', 'striped',
                  'brown', 'toxic', 'siberian', 'musical']
    return f"{random.choice(adjectives)}-{random.choice(animals)}"


